#!/usr/bin/env python3
"""
GGUF 모델 검증셋 전체 평가.

gguf_inference.py의 --test 모드는 훈련 데이터 20~50개만 확인.
이 스크립트는 competition_val_clean.csv 또는 data/processed/validation.json으로
전체 검증셋 정확도를 측정함.

사용법:
  python eval_gguf.py \\
      --model outputs/quantized/model_q3km.gguf \\
      --val_json data/processed/validation.json \\
      --n_gpu_layers -1

  # 세션 누수 없는 val 사용 (있을 경우):
  python eval_gguf.py \\
      --model outputs/quantized/model_q3km.gguf \\
      --val_jsonl data_alt/train.jsonl \\
      --val_labels data_alt/train_labels.csv \\
      --max_samples 2000
"""

import json
import csv
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

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


def build_qwen_prompt(context: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{context}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def parse_action(output: str) -> str:
    cleaned = output.strip().lower().split()[0] if output.strip() else ""
    if cleaned in ACTIONS:
        return cleaned
    for action in ACTIONS:
        if action in cleaned:
            return action
    keyword_map = {
        "read": "read_file", "write": "write_file", "edit": "edit_file",
        "patch": "apply_patch", "grep": "grep_search", "glob": "glob_pattern",
        "list": "list_directory", "dir": "list_directory",
        "bash": "run_bash", "shell": "run_bash",
        "test": "run_tests", "lint": "lint_or_typecheck",
        "web": "web_search", "ask": "ask_user",
        "plan": "plan_task", "respond": "respond_only",
    }
    for kw, action in keyword_map.items():
        if kw in cleaned:
            return action
    return "respond_only"


def load_val_from_json(json_path: str, max_samples: int = None):
    """data/processed/validation.json 로드 → (contexts, labels) 리스트."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    if max_samples:
        data = data[:max_samples]

    contexts, labels = [], []
    for item in data:
        msgs = item.get("messages", [])
        user_content = next((m["content"] for m in msgs if m["role"] == "user"), "")
        label = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        contexts.append(user_content)
        labels.append(label)
    return contexts, labels


def load_val_from_jsonl(jsonl_path: str, labels_csv: str, max_samples: int = None):
    """
    data_alt/train.jsonl + train_labels.csv 로드.
    세션 누수 없는 평가에 사용.
    """
    # 레이블 로드
    labels_map = {}
    with open(labels_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels_map[row["id"]] = row["action"]

    # JSONL 로드
    items = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                if item["id"] in labels_map:
                    items.append(item)

    # 90/10 분할의 10% (validation 부분만)
    split_idx = int(len(items) * 0.9)
    val_items = items[split_idx:]
    if max_samples:
        val_items = val_items[:max_samples]

    from gguf_inference import format_context
    contexts, labels = [], []
    for item in val_items:
        contexts.append(format_context(item))
        labels.append(labels_map[item["id"]])

    return contexts, labels


def evaluate(llm, contexts: List[str], labels: List[str], batch_report_every: int = 100):
    """GGUF 모델로 정확도 평가."""
    from tqdm import tqdm

    correct = 0
    preds = []
    errors = []
    t0 = time.time()

    for i, (ctx, label) in enumerate(tqdm(zip(contexts, labels), total=len(contexts), desc="평가")):
        prompt = build_qwen_prompt(ctx)
        result = llm(
            prompt,
            max_tokens=16,
            temperature=0.0,
            stop=["<|im_end|>", "<|im_start|>"],
            echo=False,
        )
        raw = result["choices"][0]["text"]
        pred = parse_action(raw)
        preds.append(pred)

        ok = (pred == label)
        if ok:
            correct += 1
        else:
            errors.append((label, pred, ctx[:80]))

        # 중간 보고
        if (i + 1) % batch_report_every == 0:
            acc_so_far = correct / (i + 1)
            elapsed = time.time() - t0
            remaining = (len(contexts) - i - 1) * (elapsed / (i + 1))
            logger.info(
                f"  [{i+1}/{len(contexts)}] 현재 정확도: {acc_so_far*100:.2f}% "
                f"| 남은 시간: {remaining/60:.1f}분"
            )

    elapsed = time.time() - t0
    acc = correct / len(labels) if labels else 0

    return acc, preds, errors, elapsed


def main():
    parser = argparse.ArgumentParser(description="GGUF 모델 검증셋 정확도 평가")
    parser.add_argument("--model", required=True, help="GGUF 파일 경로")
    parser.add_argument("--val_json", default="data/processed/validation.json",
                        help="validation.json (chat 형식)")
    parser.add_argument("--val_jsonl", default=None,
                        help="train.jsonl (원본, 선택적)")
    parser.add_argument("--val_labels", default="data_alt/train_labels.csv",
                        help="train_labels.csv (val_jsonl 사용 시 필요)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="평가 샘플 수 제한")
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    parser.add_argument("--n_ctx", type=int, default=2048)
    parser.add_argument("--output_csv", default=None,
                        help="오류 케이스 저장 CSV (선택)")
    args = parser.parse_args()

    # GGUF 로드
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError("pip install llama-cpp-python")

    logger.info(f"GGUF 로드: {args.model}")
    llm = Llama(
        model_path=args.model,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        n_threads=4,
        verbose=False,
    )
    logger.info("✓ 로드 완료")

    # 검증 데이터 로드
    if args.val_jsonl and Path(args.val_jsonl).exists():
        logger.info(f"검증 데이터: {args.val_jsonl} + {args.val_labels}")
        contexts, labels = load_val_from_jsonl(
            args.val_jsonl, args.val_labels, args.max_samples
        )
    elif Path(args.val_json).exists():
        logger.info(f"검증 데이터: {args.val_json}")
        contexts, labels = load_val_from_json(args.val_json, args.max_samples)
    else:
        raise FileNotFoundError(
            f"{args.val_json} 없음. --val_jsonl 또는 --val_json 경로 확인."
        )

    logger.info(f"검증 샘플 수: {len(labels)}")
    label_dist = Counter(labels)
    logger.info(f"레이블 분포 (상위 5): {dict(label_dist.most_common(5))}")

    # 평가 실행
    logger.info("\n평가 시작...")
    acc, preds, errors, elapsed = evaluate(llm, contexts, labels)

    # 결과 출력
    logger.info("\n" + "="*60)
    logger.info(f"최종 정확도: {acc*100:.2f}% ({sum(1 for p, l in zip(preds, labels) if p==l)}/{len(labels)})")
    logger.info(f"소요 시간  : {elapsed:.1f}초 ({elapsed/len(labels)*1000:.0f}ms/샘플)")
    logger.info("="*60)

    # 클래스별 정확도
    logger.info("\n클래스별 정확도:")
    class_correct = Counter()
    class_total = Counter()
    for pred, label in zip(preds, labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1

    for action in ACTIONS:
        total = class_total[action]
        if total > 0:
            class_acc = class_correct[action] / total
            bar = "█" * int(class_acc * 20) + "░" * (20 - int(class_acc * 20))
            logger.info(f"  {action:20s}: {class_acc*100:5.1f}% {bar} ({class_total[action]:4d}개)")

    # 오류 케이스
    logger.info(f"\n오류 케이스 (상위 10개):")
    for i, (label, pred, ctx_snippet) in enumerate(errors[:10]):
        logger.info(f"  [{i+1}] 정답={label}, 예측={pred}")
        logger.info(f"       컨텍스트: {ctx_snippet}...")

    # 혼동 행렬 (주요 클래스)
    from collections import defaultdict
    confusion = defaultdict(Counter)
    for pred, label in zip(preds, labels):
        if pred != label:
            confusion[label][pred] += 1

    logger.info("\n주요 혼동 패턴 (정답→오예측):")
    top_confusions = sorted(
        [(label, pred, cnt) for label, preds_cnt in confusion.items()
         for pred, cnt in preds_cnt.most_common(2)],
        key=lambda x: -x[2]
    )[:10]
    for label, pred, cnt in top_confusions:
        logger.info(f"  {label:20s} → {pred:20s}: {cnt}회")

    # CSV 저장
    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "pred", "context_snippet"])
            for label, pred, ctx_snippet in errors:
                writer.writerow([label, pred, ctx_snippet])
        logger.info(f"\n✓ 오류 케이스 저장: {args.output_csv}")

    # 다음 단계 가이드
    logger.info(f"\n다음 단계:")
    if acc >= 0.75:
        logger.info("  ✅ 75% 이상 → Budget Router 실행 준비 완료")
        logger.info("     python extract_lgbm_probs.py ...")
        logger.info("     python budget_router.py ...")
    elif acc >= 0.65:
        logger.info("  ⚠️  65~75% → 제출 가능하나 추가 학습 고려")
    else:
        logger.info("  ❌ 65% 미만 → 재학습 필요 (sbatch submit_retrain.sh)")


if __name__ == "__main__":
    main()
