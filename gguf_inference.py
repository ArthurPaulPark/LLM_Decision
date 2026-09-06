#!/usr/bin/env python3
"""
GGUF 모델 추론 스크립트 (Qwen2.5 전용)

BF16 모델 테스트에서 4/5 정확도 확인 → 훈련 정상.
이전 게이트 평가 실패 원인: 잘못된 프롬프트 템플릿.
이 스크립트는 훈련과 동일한 Qwen ChatML 템플릿을 사용.

사용법:
    # 단독 테스트
    python gguf_inference.py --model outputs/quantized/model_q3km.gguf --test

    # 대회 제출용 예측
    python gguf_inference.py \
        --model outputs/quantized/model_q3km.gguf \
        --test_file data_alt/test.jsonl \
        --output submission_gguf.csv

    # LightGBM과 앙상블 (hybrid)
    python gguf_inference.py \
        --model outputs/quantized/model_q3km.gguf \
        --test_file data_alt/test.jsonl \
        --lgbm_csv submission.csv \
        --alpha 0.3 \
        --output submission_hybrid.csv
"""

import json
import csv
import argparse
import logging
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# ── 14개 액션 ──────────────────────────────────────────────────────────────────
ACTIONS = [
    "read_file", "write_file", "edit_file", "apply_patch",
    "grep_search", "glob_pattern", "list_directory",
    "run_bash", "run_tests", "lint_or_typecheck",
    "web_search", "ask_user", "plan_task", "respond_only",
]

SYSTEM_PROMPT = """You are an AI coding assistant decision model.
Given a coding session context, output exactly one action label.

Available actions:
- read_file: Open and read a specific file
- write_file: Create a new file
- edit_file: Modify an existing file
- apply_patch: Apply a multi-file patch
- grep_search: Search for a pattern in code
- glob_pattern: Find files matching a pattern
- list_directory: List directory contents
- run_bash: Execute a shell command
- run_tests: Run test suite
- lint_or_typecheck: Run linter or type checker
- web_search: Search the web for information
- ask_user: Ask the user a clarifying question
- plan_task: Create a step-by-step plan before acting
- respond_only: Reply with text only, no tool use needed

Output only the action label, nothing else."""


# ── Qwen ChatML 템플릿 (훈련과 동일) ────────────────────────────────────────────
def build_qwen_prompt(context: str) -> str:
    """
    훈련에 사용된 것과 동일한 Qwen ChatML 포맷.
    <|im_start|>system / user / assistant 구조.
    """
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{context}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ── 세션 컨텍스트 포맷 (competition_adapter.py와 동일) ────────────────────────
def format_context(item: Dict[str, Any]) -> str:
    meta = item["session_meta"]
    workspace = meta.get("workspace", {})
    lang_mix = workspace.get("language_mix", {})
    top_lang = max(lang_mix, key=lang_mix.get) if lang_mix else "unknown"

    parts = [
        "[Session Context]",
        f"User tier: {meta.get('user_tier', 'unknown')}",
        f"Language preference: {meta.get('language_pref', 'unknown')}",
        f"Primary language: {top_lang} ({lang_mix.get(top_lang, 0)*100:.0f}%)",
        f"Codebase size: {workspace.get('loc', 0):,} lines",
        f"Git dirty: {workspace.get('git_dirty', False)}",
        f"CI status: {workspace.get('last_ci_status', 'none')}",
        f"Open files: {', '.join(workspace.get('open_files', [])) or 'none'}",
        f"Budget tokens remaining: {meta.get('budget_tokens_remaining', 0):,}",
        f"Turn index: {meta.get('turn_index', 0)}",
    ]

    history = item.get("history", [])
    if history:
        parts.append("\n[Conversation History]")
        for turn in history[-6:]:
            role = turn.get("role", "")
            if role == "user":
                parts.append(f"User: {turn.get('content', '')[:200]}")
            elif role == "assistant_action":
                name = turn.get("name", "")
                summary = turn.get("result_summary", "")[:100]
                parts.append(f"Assistant action: {name} → {summary}")

    parts.append("\n[Current User Request]")
    parts.append(item.get("current_prompt", ""))
    return "\n".join(parts)


# ── 출력 파싱 ─────────────────────────────────────────────────────────────────
def parse_action(output: str) -> str:
    cleaned = output.strip().lower().split()[0] if output.strip() else ""

    if cleaned in ACTIONS:
        return cleaned
    for action in ACTIONS:
        if action in cleaned:
            return action

    keyword_map = {
        "read": "read_file", "open": "read_file",
        "write": "write_file", "creat": "write_file",
        "edit": "edit_file", "modif": "edit_file",
        "patch": "apply_patch",
        "grep": "grep_search", "search": "grep_search",
        "glob": "glob_pattern", "find": "glob_pattern",
        "list": "list_directory", "dir": "list_directory",
        "bash": "run_bash", "shell": "run_bash", "execut": "run_bash",
        "test": "run_tests",
        "lint": "lint_or_typecheck", "type": "lint_or_typecheck",
        "web": "web_search",
        "ask": "ask_user",
        "plan": "plan_task",
        "respond": "respond_only", "reply": "respond_only",
    }
    for kw, action in keyword_map.items():
        if kw in cleaned:
            return action

    logger.debug(f"Unknown output: '{output[:30]}' → respond_only")
    return "respond_only"


# ── GGUF 모델 로드 ────────────────────────────────────────────────────────────
def load_gguf(model_path: str, n_ctx: int = 2048):
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError("pip install llama-cpp-python")

    logger.info(f"GGUF 모델 로드: {model_path}")
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=8,
        verbose=False,
    )
    logger.info("✓ 로드 완료")
    return llm


# ── 단일 예측 ─────────────────────────────────────────────────────────────────
def predict_one(llm, context: str) -> str:
    prompt = build_qwen_prompt(context)
    result = llm(
        prompt,
        max_tokens=16,
        temperature=0.0,          # greedy
        stop=["<|im_end|>", "<|im_start|>"],  # Qwen 종료 토큰
        echo=False,
    )
    raw = result["choices"][0]["text"]
    return parse_action(raw)


# ── 배치 예측 ─────────────────────────────────────────────────────────────────
def predict_batch(llm, items: List[Dict], batch_size: int = 1) -> List[str]:
    """llama-cpp는 배치 미지원 → 순차 처리."""
    predictions = []
    for item in tqdm(items, desc="Predicting"):
        context = format_context(item)
        pred = predict_one(llm, context)
        predictions.append(pred)
    return predictions


# ── 빠른 정확도 테스트 ──────────────────────────────────────────────────────────
def run_test(llm, n: int = 20):
    """훈련 데이터 n개로 빠른 정확도 확인."""
    import json as _json
    train_path = Path("data/processed/train.json")
    if not train_path.exists():
        train_path = Path("data_alt/train.jsonl")
        with open(train_path) as f:
            data = [_json.loads(l) for l in f if l.strip()][:n]
        # train.jsonl은 라벨 없음 → 학습 JSON 사용 불가
        logger.warning("data/processed/train.json이 없어 정확도 측정 불가")
        return

    with open(train_path) as f:
        data = _json.load(f)[:n]

    correct = 0
    dist = Counter()
    for i, sample in enumerate(data):
        label = sample["messages"][2]["content"]
        context_parts = []
        for msg in sample["messages"][:2]:
            if msg["role"] == "user":
                context_parts.append(msg["content"])
        context = context_parts[0] if context_parts else ""

        prompt = build_qwen_prompt(context)
        result = llm(prompt, max_tokens=16, temperature=0.0,
                     stop=["<|im_end|>", "<|im_start|>"], echo=False)
        raw = result["choices"][0]["text"]
        pred = parse_action(raw)
        ok = pred == label
        correct += ok
        dist[pred] += 1
        logger.info(f"[{i+1:2d}] pred={pred:20s} | label={label:20s} | {'✓' if ok else '✗'}")

    logger.info(f"\n정확도: {correct}/{n} ({correct/n*100:.1f}%)")
    logger.info("예측 분포:")
    for k, v in dist.most_common():
        logger.info(f"  {k:20s}: {v}")


# ── LightGBM + LLM 앙상블 ─────────────────────────────────────────────────────
def hybrid_predict(
    llm_preds: List[str],
    lgbm_csv: str,
    ids: List[str],
    alpha: float = 0.3,
) -> List[str]:
    """
    alpha: LLM 가중치 (0=LightGBM 단독, 1=LLM 단독)
    LLM이 확신 있는 경우(특정 조건)만 오버라이드하는 단순 rule 사용.
    """
    lgbm_preds = {}
    with open(lgbm_csv) as f:
        for row in csv.DictReader(f):
            lgbm_preds[row["id"]] = row["action"]

    final = []
    agree = 0
    for id_, llm_p in zip(ids, llm_preds):
        lgbm_p = lgbm_preds.get(id_, "respond_only")
        if llm_p == lgbm_p:
            final.append(llm_p)
            agree += 1
        elif alpha >= 0.5:
            final.append(llm_p)
        else:
            final.append(lgbm_p)

    logger.info(f"LLM-LightGBM 일치율: {agree}/{len(ids)} ({agree/len(ids)*100:.1f}%)")
    return final


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="GGUF 파일 경로")
    parser.add_argument("--test_file", default="data_alt/test.jsonl")
    parser.add_argument("--output", default="submission_gguf.csv")
    parser.add_argument("--lgbm_csv", default=None, help="LightGBM 예측 CSV (앙상블용)")
    parser.add_argument("--alpha", type=float, default=0.3, help="LLM 가중치 (0~1)")
    parser.add_argument("--test", action="store_true", help="훈련 샘플 20개 정확도 빠른 테스트")
    parser.add_argument("--n_test", type=int, default=20)
    parser.add_argument("--n_ctx", type=int, default=2048)
    args = parser.parse_args()

    llm = load_gguf(args.model, n_ctx=args.n_ctx)

    if args.test:
        run_test(llm, n=args.n_test)
        return

    # 테스트 데이터 로드
    items = []
    with open(args.test_file) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    logger.info(f"테스트 샘플: {len(items)}개")

    # 예측
    llm_preds = predict_batch(llm, items)
    ids = [item["id"] for item in items]

    # 앙상블 여부
    if args.lgbm_csv:
        final_preds = hybrid_predict(llm_preds, args.lgbm_csv, ids, alpha=args.alpha)
        logger.info(f"Hybrid (alpha={args.alpha}) 예측 완료")
    else:
        final_preds = llm_preds

    # CSV 저장
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "action"])
        for id_, pred in zip(ids, final_preds):
            writer.writerow([id_, pred])

    logger.info(f"✓ {args.output} 저장 ({len(final_preds)}개)")

    dist = Counter(final_preds)
    logger.info("예측 분포:")
    for k, v in dist.most_common():
        logger.info(f"  {k:20s}: {v:5d} ({v/len(final_preds)*100:.1f}%)")


if __name__ == "__main__":
    main()
