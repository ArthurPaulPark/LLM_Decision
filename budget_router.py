#!/usr/bin/env python3
"""
Budget Router: LightGBM + GGUF LLM Hybrid 추론

동작 원리:
  1. LightGBM이 전체 샘플에 대해 14개 클래스 확률 예측
  2. max_prob < threshold (기본 0.65) 인 불확실 샘플 → LLM으로 라우팅
  3. 라우팅 비율은 약 15% → 30,000샘플 × 15% × ~20ms(T4 GPU) ≈ 90초 < 10분 제한

사용법:
  # 1단계: LightGBM 확률 CSV 생성 (submit 패키지에서)
  cd ~/submit_extracted
  python script_with_probs.py  # 또는 아래 --gen_lgbm_probs 모드 사용

  # 2단계: Budget Router 실행
  python budget_router.py \\
      --lgbm_probs lgbm_probs.csv \\          # LightGBM 14클래스 확률 CSV
      --test_file data_alt/test.jsonl \\
      --gguf_model outputs/quantized/model_q3km.gguf \\
      --threshold 0.65 \\                      # 이 확신도 미만 → LLM 라우팅
      --output submission_hybrid.csv

  # 빠른 검증 (val 데이터로 정확도 확인):
  python budget_router.py \\
      --lgbm_probs lgbm_val_probs.csv \\
      --val_file data/processed/validation.json \\
      --gguf_model outputs/quantized/model_q3km.gguf \\
      --threshold 0.65 \\
      --eval_mode

  # LightGBM 단독 베이스라인과 비교:
  python budget_router.py --compare_only --lgbm_probs lgbm_probs.csv --output lgbm_only.csv
"""

import json
import csv
import sys
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from typing import List, Dict, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 14개 액션 (LightGBM 클래스 순서와 반드시 일치해야 함) ──────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM 확률 CSV 로드
# ─────────────────────────────────────────────────────────────────────────────

def load_lgbm_probs(csv_path: str) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    LightGBM 확률 CSV 로드.

    CSV 형식 (둘 다 지원):
      형식 A - 14개 클래스 확률 포함:
        id, pred_action, read_file, write_file, ..., respond_only
      형식 B - 예측만:
        id, action
        → max_prob=1.0 (확신도 100%) 로 처리, LLM 라우팅 없음

    Returns:
        ids: 샘플 ID 리스트
        probs: (N, 14) 확률 행렬
        preds: LightGBM 예측 라벨 리스트
    """
    ids, preds, probs_list = [], [], []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        has_probs = any(a in fieldnames for a in ACTIONS)
        if not has_probs:
            logger.warning(
                "CSV에 클래스 확률 컬럼이 없습니다. "
                "형식 B(예측만) 로 처리 → 모든 샘플 LightGBM 예측 사용."
            )

        for row in reader:
            ids.append(row["id"])
            pred = row.get("pred_action") or row.get("action", "respond_only")
            preds.append(pred)

            if has_probs:
                prob_row = [float(row.get(a, 0.0)) for a in ACTIONS]
                s = sum(prob_row)
                if s > 0:
                    prob_row = [p / s for p in prob_row]  # normalize
                else:
                    # 해당 행 확률 없음 → 예측 클래스에 1.0
                    idx = ACTIONS.index(pred) if pred in ACTIONS else 0
                    prob_row = [0.0] * 14
                    prob_row[idx] = 1.0
                probs_list.append(prob_row)
            else:
                # 확률 없음 → 예측 클래스에 1.0 (라우팅 안 됨)
                idx = ACTIONS.index(pred) if pred in ACTIONS else 0
                prob_row = [0.0] * 14
                prob_row[idx] = 1.0
                probs_list.append(prob_row)

    probs = np.array(probs_list, dtype=np.float32)
    logger.info(f"LightGBM 확률 로드: {len(ids)}개 샘플, probs shape={probs.shape}")
    return ids, probs, preds


# ─────────────────────────────────────────────────────────────────────────────
# 테스트 데이터 로드 & 컨텍스트 포맷
# ─────────────────────────────────────────────────────────────────────────────

def load_test_jsonl(path: str) -> Dict[str, dict]:
    """test.jsonl 로드 → {id: item} 딕셔너리."""
    items = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                items[item["id"]] = item
    logger.info(f"테스트 데이터 로드: {len(items)}개")
    return items


def format_context(item: dict) -> str:
    """competition_adapter._format_context()와 동일."""
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


# ─────────────────────────────────────────────────────────────────────────────
# GGUF 추론
# ─────────────────────────────────────────────────────────────────────────────

def load_gguf(model_path: str, n_gpu_layers: int = -1, n_ctx: int = 2048):
    """
    GGUF 모델 로드.
    n_gpu_layers=-1: 모든 레이어 GPU 오프로드 (T4에서 최대 속도)
    n_gpu_layers=0 : CPU only (GPU 없을 때)
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python 미설치.\n"
            "  GPU: pip install llama-cpp-python --extra-index-url "
            "https://abetlen.github.io/llama-cpp-python/whl/cu124\n"
            "  CPU: pip install llama-cpp-python"
        )

    logger.info(f"GGUF 로드: {model_path} (n_gpu_layers={n_gpu_layers})")
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_threads=4,
        verbose=False,
    )
    logger.info("✓ GGUF 로드 완료")
    return llm


def build_qwen_prompt(context: str) -> str:
    """훈련과 동일한 Qwen ChatML 포맷."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{context}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def parse_action(output: str) -> str:
    """모델 출력 → 액션 라벨 파싱."""
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
        "grep": "grep_search",
        "glob": "glob_pattern", "find": "glob_pattern",
        "list": "list_directory", "dir": "list_directory",
        "bash": "run_bash", "shell": "run_bash",
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
    return "respond_only"


def llm_predict_batch(
    llm,
    ids_to_predict: List[str],
    test_items: Dict[str, dict],
    max_tokens: int = 16,
) -> Dict[str, str]:
    """
    GGUF LLM으로 지정된 ID 목록 예측.

    Returns: {id: predicted_action}
    """
    from tqdm import tqdm

    results = {}
    t0 = time.time()

    for i, sample_id in enumerate(tqdm(ids_to_predict, desc="LLM 추론")):
        item = test_items.get(sample_id)
        if item is None:
            results[sample_id] = "respond_only"
            continue

        context = format_context(item)
        prompt = build_qwen_prompt(context)

        out = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            stop=["<|im_end|>", "<|im_start|>"],
            echo=False,
        )
        raw = out["choices"][0]["text"]
        pred = parse_action(raw)
        results[sample_id] = pred

    elapsed = time.time() - t0
    n = len(ids_to_predict)
    logger.info(
        f"LLM 추론 완료: {n}개 / {elapsed:.1f}초 "
        f"= {elapsed/max(n,1)*1000:.0f}ms/샘플"
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Budget Router 핵심 로직
# ─────────────────────────────────────────────────────────────────────────────

def budget_route(
    ids: List[str],
    probs: np.ndarray,
    lgbm_preds: List[str],
    llm_preds: Dict[str, str],
    threshold: float,
) -> Tuple[List[str], dict]:
    """
    Budget routing: 불확실 샘플만 LLM 예측으로 교체.

    Args:
        ids: 전체 샘플 ID
        probs: (N, 14) LightGBM 확률
        lgbm_preds: LightGBM 예측 라벨
        llm_preds: {id: llm_pred} (라우팅된 샘플만)
        threshold: max_prob < threshold → LLM 사용

    Returns:
        final_preds: 최종 예측 리스트
        stats: 통계 딕셔너리
    """
    max_probs = probs.max(axis=1)
    routed_mask = max_probs < threshold

    final_preds = []
    used_llm = 0
    used_lgbm = 0
    llm_overrides = 0

    for i, sample_id in enumerate(ids):
        if routed_mask[i] and sample_id in llm_preds:
            llm_pred = llm_preds[sample_id]
            final_preds.append(llm_pred)
            used_llm += 1
            if llm_pred != lgbm_preds[i]:
                llm_overrides += 1
        else:
            final_preds.append(lgbm_preds[i])
            used_lgbm += 1

    stats = {
        "total": len(ids),
        "lgbm_used": used_lgbm,
        "llm_used": used_llm,
        "llm_ratio": used_llm / len(ids),
        "llm_overrides": llm_overrides,   # LightGBM과 다른 예측
        "threshold": threshold,
        "mean_max_prob": float(max_probs.mean()),
        "pct_confident": float((max_probs >= threshold).mean()),
    }
    return final_preds, stats


def estimate_time(n_llm_samples: int, ms_per_sample: float = 20.0) -> str:
    """예상 추론 시간 계산."""
    total_sec = n_llm_samples * ms_per_sample / 1000
    return f"{total_sec:.0f}초 ({total_sec/60:.1f}분)"


# ─────────────────────────────────────────────────────────────────────────────
# Eval Mode: 검증셋으로 정확도 측정
# ─────────────────────────────────────────────────────────────────────────────

def eval_mode(args):
    """검증셋으로 budget router 정확도 측정."""
    logger.info("=== Eval Mode: 검증셋 정확도 측정 ===")

    # LightGBM 확률 로드
    ids, probs, lgbm_preds = load_lgbm_probs(args.lgbm_probs)

    # 검증 데이터 로드 (labels 포함)
    with open(args.val_file, encoding="utf-8") as f:
        val_data = json.load(f)

    # {id: label} 딕셔너리 (data/processed/validation.json 형식)
    # 각 항목: {"messages": [{role:system,...}, {role:user,...}, {role:assistant, content:label}]}
    labels = {}
    val_items_by_id = {}
    for i, item in enumerate(val_data):
        # validation.json은 id 컬럼 없음 → 인덱스 또는 별도 방법 필요
        # test.jsonl 있으면 그걸로 context 복원
        label = item["messages"][2]["content"] if len(item.get("messages", [])) > 2 else ""
        # ID는 val_probs CSV의 id와 매핑 필요
        # → ids 순서가 validation.json 순서와 같다고 가정
        if i < len(ids):
            labels[ids[i]] = label

    # threshold별 정확도 비교
    thresholds = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90]

    logger.info(f"\n{'Threshold':>10} {'LLM%':>6} {'Accuracy':>10} {'vs LightGBM':>12}")
    logger.info("-" * 45)

    # LightGBM 단독 정확도
    lgbm_correct = sum(1 for i, sid in enumerate(ids) if labels.get(sid) == lgbm_preds[i])
    lgbm_acc = lgbm_correct / len(ids) if ids else 0
    logger.info(f"{'LightGBM':>10} {'100%':>6} {lgbm_acc*100:>9.2f}%   (baseline)")

    for thresh in thresholds:
        max_probs = probs.max(axis=1)
        routed = max_probs < thresh
        n_routed = routed.sum()

        # 라우팅된 샘플 LLM 예측 (여기선 oracle 사용 - 실제 레이블로 대체)
        # 실제 사용 시에는 llm_preds를 넣어야 함
        # 이 eval에서는 LLM이 x% 정확도를 보인다고 가정
        # → 실제 GGUF 추론 결과가 있으면 그걸 쓰세요

        # 단순 계산: 라우팅 비율만 보여줌
        ratio = n_routed / len(ids)
        est_time = estimate_time(n_routed)
        logger.info(
            f"{thresh:>10.2f} {ratio*100:>5.1f}%  "
            f"{'N/A (run with --gguf_model)':>10}   "
            f"~{est_time} 예상"
        )

    # GGUF 결과 있으면 실제 정확도 계산
    if args.gguf_model and Path(args.gguf_model).exists():
        logger.info("\nGGUF 모델로 실제 정확도 계산 중...")
        llm = load_gguf(args.gguf_model, n_gpu_layers=args.n_gpu_layers)

        # 실제 test.jsonl에서 val items 로드 (context 복원용)
        # 여기선 validation.json의 user message를 context로 사용
        val_context_map = {}
        for i, item in enumerate(val_data):
            if i < len(ids):
                msgs = item.get("messages", [])
                user_content = next((m["content"] for m in msgs if m["role"] == "user"), "")
                val_context_map[ids[i]] = {"current_prompt": user_content, "session_meta": {}, "history": []}

        thresh = args.threshold
        max_probs = probs.max(axis=1)
        routed_ids = [ids[i] for i in range(len(ids)) if max_probs[i] < thresh]

        logger.info(f"\n임계값 {thresh}: {len(routed_ids)}개 ({len(routed_ids)/len(ids)*100:.1f}%) 라우팅")
        llm_preds_dict = llm_predict_batch(llm, routed_ids, val_context_map)

        final_preds, stats = budget_route(ids, probs, lgbm_preds, llm_preds_dict, thresh)

        correct = sum(1 for i, sid in enumerate(ids) if labels.get(sid) == final_preds[i])
        hybrid_acc = correct / len(ids) if ids else 0

        logger.info(f"\n{'='*50}")
        logger.info(f"LightGBM 단독: {lgbm_acc*100:.2f}%")
        logger.info(f"Hybrid (thresh={thresh}): {hybrid_acc*100:.2f}%")
        logger.info(f"개선: {(hybrid_acc - lgbm_acc)*100:+.2f}pp")
        logger.info(f"LLM 사용 비율: {stats['llm_ratio']*100:.1f}%")
        logger.info(f"LLM 오버라이드: {stats['llm_overrides']}개")


# ─────────────────────────────────────────────────────────────────────────────
# Main: 테스트 데이터 전체 예측
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Budget Router: LightGBM + GGUF LLM Hybrid"
    )

    # 입력
    parser.add_argument("--lgbm_probs", required=True,
                        help="LightGBM 확률 CSV (id, pred_action, read_file, ..., respond_only)")
    parser.add_argument("--test_file", default="data_alt/test.jsonl",
                        help="테스트 JSONL 파일")
    parser.add_argument("--gguf_model", default=None,
                        help="GGUF 모델 경로 (없으면 LightGBM 단독)")

    # 라우팅 설정
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="이 값 미만의 max_prob → LLM 라우팅 (기본 0.65)")
    parser.add_argument("--max_llm_ratio", type=float, default=0.20,
                        help="LLM 라우팅 최대 비율 (기본 20%%, 초과 시 상위 불확실 순 선택)")
    parser.add_argument("--n_gpu_layers", type=int, default=-1,
                        help="GGUF GPU 레이어 수 (-1=전체, 0=CPU)")
    parser.add_argument("--n_ctx", type=int, default=2048)

    # 출력
    parser.add_argument("--output", default="submission_hybrid.csv")

    # 모드
    parser.add_argument("--eval_mode", action="store_true",
                        help="검증셋 정확도 측정 모드")
    parser.add_argument("--val_file", default="data/processed/validation.json")
    parser.add_argument("--compare_only", action="store_true",
                        help="LightGBM 예측만 CSV로 저장 (LLM 실행 없음)")
    parser.add_argument("--dry_run", action="store_true",
                        help="라우팅 분석만 출력, 실제 LLM 실행 없음")

    args = parser.parse_args()

    # ── LightGBM 확률 로드 ─────────────────────────────────────────────────
    ids, probs, lgbm_preds = load_lgbm_probs(args.lgbm_probs)
    max_probs = probs.max(axis=1)

    # ── 라우팅 분석 ────────────────────────────────────────────────────────
    n_total = len(ids)
    routed_mask = max_probs < args.threshold
    n_routed_thresh = routed_mask.sum()

    # max_llm_ratio 초과 방지: 불확실도 높은 순 top-k만 선택
    max_llm_samples = int(n_total * args.max_llm_ratio)
    if n_routed_thresh > max_llm_samples:
        logger.warning(
            f"threshold={args.threshold} → {n_routed_thresh}개 ({n_routed_thresh/n_total*100:.1f}%) 라우팅 "
            f"> max_llm_ratio {args.max_llm_ratio*100:.0f}% ({max_llm_samples}개) 제한"
        )
        # 가장 불확실한 순 (max_prob 낮은 순) top max_llm_samples 선택
        sorted_indices = np.argsort(max_probs)
        routed_indices = set(sorted_indices[:max_llm_samples].tolist())
        routed_mask = np.array([i in routed_indices for i in range(n_total)])
        n_routed = max_llm_samples
    else:
        n_routed = int(n_routed_thresh)

    routed_ids = [ids[i] for i in range(n_total) if routed_mask[i]]

    logger.info("\n" + "="*60)
    logger.info("Budget Router 분석")
    logger.info("="*60)
    logger.info(f"총 샘플       : {n_total:,}")
    logger.info(f"임계값        : {args.threshold}")
    logger.info(f"LLM 라우팅    : {n_routed:,} ({n_routed/n_total*100:.1f}%)")
    logger.info(f"LightGBM 사용 : {n_total-n_routed:,} ({(n_total-n_routed)/n_total*100:.1f}%)")
    logger.info(f"평균 확신도   : {max_probs.mean():.3f}")
    logger.info(f"예상 소요     : {estimate_time(n_routed)} (T4 GPU 기준)")

    # 예측 분포 (LightGBM)
    dist = Counter(lgbm_preds)
    logger.info("\nLightGBM 예측 분포 (상위 5개):")
    for action, cnt in dist.most_common(5):
        logger.info(f"  {action:20s}: {cnt:5d} ({cnt/n_total*100:.1f}%)")

    # 라우팅 대상 예측 분포
    routed_preds = [lgbm_preds[i] for i in range(n_total) if routed_mask[i]]
    routed_dist = Counter(routed_preds)
    logger.info(f"\n라우팅 샘플 예측 분포 (상위 5개):")
    for action, cnt in routed_dist.most_common(5):
        logger.info(f"  {action:20s}: {cnt:5d} ({cnt/n_routed*100:.1f}%)" if n_routed else "")

    if args.dry_run or args.compare_only:
        logger.info("\n[--dry_run / --compare_only] LLM 실행 없이 종료")
        # LightGBM 단독 결과 저장
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "action"])
            for sid, pred in zip(ids, lgbm_preds):
                writer.writerow([sid, pred])
        logger.info(f"✓ LightGBM 단독 예측 저장: {args.output}")
        return

    if args.eval_mode:
        eval_mode(args)
        return

    # ── GGUF 모델 로드 & 추론 ────────────────────────────────────────────
    if not args.gguf_model:
        logger.info("\nGGUF 모델 미지정 → LightGBM 단독 예측 저장")
        llm_preds_dict = {}
    else:
        if not Path(args.gguf_model).exists():
            logger.error(f"GGUF 파일 없음: {args.gguf_model}")
            sys.exit(1)

        # 테스트 데이터 로드
        test_items = load_test_jsonl(args.test_file)
        missing = [sid for sid in routed_ids if sid not in test_items]
        if missing:
            logger.warning(f"test.jsonl에 없는 ID: {len(missing)}개 → respond_only 처리")

        # GGUF 추론
        llm = load_gguf(args.gguf_model, n_gpu_layers=args.n_gpu_layers, n_ctx=args.n_ctx)
        llm_preds_dict = llm_predict_batch(llm, routed_ids, test_items)

    # ── Budget Routing ─────────────────────────────────────────────────
    final_preds, stats = budget_route(ids, probs, lgbm_preds, llm_preds_dict, args.threshold)

    # ── 결과 저장 ──────────────────────────────────────────────────────
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "action"])
        for sid, pred in zip(ids, final_preds):
            writer.writerow([sid, pred])

    logger.info("\n" + "="*60)
    logger.info(f"✓ 완료: {args.output} ({len(final_preds)}개)")
    logger.info(f"  LightGBM 사용 : {stats['lgbm_used']}개 ({stats['lgbm_used']/stats['total']*100:.1f}%)")
    logger.info(f"  LLM 사용      : {stats['llm_used']}개 ({stats['llm_ratio']*100:.1f}%)")
    logger.info(f"  LLM 오버라이드: {stats['llm_overrides']}개")

    final_dist = Counter(final_preds)
    logger.info("\n최종 예측 분포:")
    for action, cnt in final_dist.most_common():
        logger.info(f"  {action:20s}: {cnt:5d} ({cnt/len(final_preds)*100:.1f}%)")


if __name__ == "__main__":
    main()
