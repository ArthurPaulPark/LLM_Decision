#!/usr/bin/env python3
"""
대회 제출용 추론 스크립트.

Fine-tuned 모델(base + LoRA adapter)로 test.jsonl 예측 후 submission.csv 생성.

사용법:
    python predict.py                                  # 기본값으로 실행
    python predict.py --adapter outputs/checkpoints   # 어댑터 경로 직접 지정
    python predict.py --batch_size 4                  # 배치 크기 조정
"""

import json
import csv
import argparse
import logging
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# ── 14개 액션 레이블 ──────────────────────────────────────────────────────────
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


# ── 데이터 전처리 (competition_adapter.py와 동일) ──────────────────────────────
def format_context(item: Dict[str, Any]) -> str:
    """세션 컨텍스트 → 프롬프트 문자열."""
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


def parse_action(output: str) -> str:
    """
    모델 출력에서 유효한 액션 레이블 추출.

    1) 정확히 일치
    2) 부분 포함
    3) 키워드 기반 fallback
    """
    raw = output.strip()
    cleaned = raw.lower().strip()

    # 1) 정확 일치
    if cleaned in ACTIONS:
        return cleaned

    # 2) 부분 포함
    for action in ACTIONS:
        if action in cleaned:
            return action

    # 3) 키워드 fallback
    keyword_map = {
        "read":    "read_file",
        "open":    "read_file",
        "write":   "write_file",
        "creat":   "write_file",
        "edit":    "edit_file",
        "modif":   "edit_file",
        "patch":   "apply_patch",
        "grep":    "grep_search",
        "search":  "grep_search",
        "glob":    "glob_pattern",
        "find":    "glob_pattern",
        "list":    "list_directory",
        "dir":     "list_directory",
        "bash":    "run_bash",
        "shell":   "run_bash",
        "execut":  "run_bash",
        "test":    "run_tests",
        "lint":    "lint_or_typecheck",
        "type":    "lint_or_typecheck",
        "web":     "web_search",
        "ask":     "ask_user",
        "plan":    "plan_task",
        "respond": "respond_only",
        "reply":   "respond_only",
    }
    for kw, action in keyword_map.items():
        if kw in cleaned:
            return action

    logging.warning(f"Unknown output: '{raw[:50]}' → fallback: respond_only")
    return "respond_only"


# ── 모델 로드 ─────────────────────────────────────────────────────────────────
def load_model(base_model: str, adapter_path: str):
    """base model + LoRA adapter 로드 (추론 전용, 4-bit 불필요)."""
    logging.info(f"Loading tokenizer: {adapter_path}")
    # 어댑터 폴더에 저장된 tokenizer 우선 사용
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # batch generation 시 왼쪽 패딩

    logging.info(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    logging.info(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        used = torch.cuda.memory_allocated() / 1024**3
        logging.info(f"✓ GPU: {torch.cuda.get_device_name(0)} | {used:.1f}/{mem:.1f} GB")

    logging.info("✓ Model ready")
    return model, tokenizer


# ── 배치 추론 ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_batch(
    model,
    tokenizer,
    items: List[Dict],
    batch_size: int = 8,
    max_input_length: int = 1024,
    max_new_tokens: int = 16,
) -> List[str]:
    """배치 추론. 결과는 items와 같은 순서."""
    predictions = []

    for i in tqdm(range(0, len(items), batch_size), desc="Predicting"):
        batch = items[i : i + batch_size]

        # chat template 적용
        texts = []
        for item in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_context(item)},
            ]
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,   # Gemma 4: 추론 토큰 비활성화
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            texts.append(text)

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,           # greedy decoding (결정론적)
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # 입력 토큰 제거 → 생성된 부분만 디코딩
        input_len = inputs["input_ids"].shape[1]
        for output_ids in outputs:
            decoded = tokenizer.decode(
                output_ids[input_len:], skip_special_tokens=True
            )
            action = parse_action(decoded)
            predictions.append(action)
            logging.debug(f"  raw='{decoded.strip()}' → {action}")

    return predictions


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="대회 제출용 추론 스크립트")
    parser.add_argument(
        "--base_model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HF 캐시 경로 또는 모델명 (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--adapter",
        default="outputs/checkpoints",
        help="LoRA 어댑터 경로 (default: outputs/checkpoints)",
    )
    parser.add_argument(
        "--test_file",
        default="data_alt/test.jsonl",
        help="테스트 JSONL 파일 (default: data_alt/test.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="submission.csv",
        help="출력 CSV 파일명 (default: submission.csv)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="배치 크기 (default: 8)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="각 예측의 raw 출력 표시",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # ── 테스트 데이터 로드 ──
    test_path = Path(args.test_file)
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    test_items = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_items.append(json.loads(line))

    logging.info(f"✓ Test examples: {len(test_items)}")

    # ── 모델 로드 ──
    model, tokenizer = load_model(args.base_model, args.adapter)

    # ── 예측 ──
    predictions = predict_batch(model, tokenizer, test_items, batch_size=args.batch_size)

    # ── CSV 저장 ──
    output_path = Path(args.output)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "action"])
        for item, pred in zip(test_items, predictions):
            writer.writerow([item["id"], pred])

    logging.info(f"✓ Saved {len(predictions)} predictions → {output_path}")

    # ── 예측 분포 출력 ──
    dist = Counter(predictions)
    logging.info("\n예측 분포:")
    for action, count in sorted(dist.items(), key=lambda x: -x[1]):
        pct = count / len(predictions) * 100
        logging.info(f"  {action:20s}: {count:5d} ({pct:.1f}%)")

    # ── 개별 예측 결과 출력 ──
    logging.info("\n개별 예측:")
    for item, pred in zip(test_items, predictions):
        prompt_preview = item.get("current_prompt", "")[:60]
        logging.info(f"  [{pred:20s}] {prompt_preview}...")


if __name__ == "__main__":
    main()
