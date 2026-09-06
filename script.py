#!/usr/bin/env python3
"""
대회 제출용 추론 스크립트 (Budget Router 버전)
평가 서버가 자동으로 실행함.

submit.zip 구조:
  model/
    lgbm_fold1.txt ~ lgbm_fold5.txt  ← LightGBM 5-fold 모델
    lib/
      feature_extractor_v2.py         ← 피처 추출기
    svd_prompt_components.npy          ← SVD 컴포넌트
    svd_history_components.npy
    feature_schema.json
    group_stats.json
    model.gguf                         ← GGUF 양자화 모델 (Budget Router)
  script.py
  requirements.txt
  data/                               ← 평가 서버가 주입
    test.jsonl
  output/                             ← 평가 서버가 생성
    submission.csv

전략 (Budget Router):
  1) feature_extractor_v2로 피처 추출
  2) LightGBM 5-fold 앙상블 확률 예측
  3) max_prob < THRESHOLD → LLM(GGUF)으로 재예측
  4) output/submission.csv 저장
"""

import os
import sys
import json
import time
import csv
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
MODEL_DIR  = SCRIPT_DIR / "model"
DATA_DIR   = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"

TEST_FILE  = DATA_DIR / "test.jsonl"
OUT_CSV    = OUTPUT_DIR / "submission.csv"
GGUF_FILE  = MODEL_DIR / "model.gguf"

# model/lib/ 를 sys.path에 추가 (feature_extractor_v2 임포트용)
sys.path.insert(0, str(MODEL_DIR / "lib"))
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

# Budget Router 파라미터
THRESHOLD     = 0.65   # LightGBM max_prob 임계값
MAX_LLM_RATIO = 0.20   # 최대 LLM 라우팅 비율 (20%)
TIME_LIMIT    = 570    # 9분 30초 (10분 - 30초 여유)

# 14개 클래스 (알파벳 순 정렬 - LightGBM 학습 시 기본 순서)
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


# ── LightGBM 5-fold 앙상블 ────────────────────────────────────────────────────
def load_lgbm_models(model_dir: Path) -> list:
    """lgbm_fold1~5.txt 로드."""
    import lightgbm as lgb
    models = []
    for i in range(1, 6):
        path = model_dir / f"lgbm_fold{i}.txt"
        if path.exists():
            booster = lgb.Booster(model_file=str(path))
            models.append(booster)
            logger.info(f"  ✓ fold{i} 로드")
    if not models:
        raise FileNotFoundError(f"{model_dir}에 lgbm_fold*.txt 없음")
    return models


def predict_lgbm(models: list, features) -> Tuple[List[str], List[List[float]]]:
    """
    5-fold 앙상블 예측.
    Returns: (preds: List[str], probs: List[List[float]])
    """
    import numpy as np
    fold_probs = []
    for model in models:
        proba = model.predict(features)  # (N, 14)
        fold_probs.append(proba)
    probs = np.mean(fold_probs, axis=0)  # (N, 14)

    # 클래스 이름 가져오기
    try:
        class_names = list(models[0].class_names())
    except Exception:
        class_names = sorted(ACTIONS)  # fallback: 알파벳 순

    pred_indices = probs.argmax(axis=1)
    preds = [class_names[i] if i < len(class_names) else "respond_only"
             for i in pred_indices]
    return preds, probs.tolist(), class_names


# ── 피처 추출 ─────────────────────────────────────────────────────────────────
def extract_features(items: List[dict], model_dir: Path):
    """
    feature_extractor_v2로 피처 추출.
    실패 시 fallback 피처 사용.
    """
    # 방법 1: feature_extractor_v2.FeatureExtractor
    try:
        from feature_extractor_v2 import FeatureExtractor
        extractor = FeatureExtractor(str(model_dir))
        features = extractor.transform(items)
        logger.info(f"  ✓ feature_extractor_v2 사용 → shape: {features.shape}")
        return features
    except ImportError:
        logger.warning("  feature_extractor_v2 없음, auto_features 시도...")
    except Exception as e:
        logger.warning(f"  feature_extractor_v2 오류: {e}, auto_features 시도...")

    # 방법 2: auto_features
    try:
        import auto_features
        features = auto_features.extract(items, str(model_dir))
        logger.info(f"  ✓ auto_features 사용 → shape: {features.shape}")
        return features
    except (ImportError, AttributeError) as e:
        logger.warning(f"  auto_features 실패: {e}")

    # 방법 3: fallback 수치 피처 (정확도 낮지만 크래시 방지)
    logger.warning("  Fallback 수치 피처 사용 (정확도 저하 가능)")
    return _fallback_features(items)


def _fallback_features(items: List[dict]):
    """feature_extractor가 없을 때 사용하는 기본 피처."""
    import numpy as np
    feat_list = []
    for item in items:
        meta = item.get("session_meta", {})
        ws   = meta.get("workspace", {})
        history = item.get("history", [])
        action_turns = [t for t in history if t.get("role") == "assistant_action"]
        prompt = item.get("current_prompt", "").lower()

        last_action = action_turns[-1].get("name", "") if action_turns else ""
        last_action_idx = ACTIONS.index(last_action) if last_action in ACTIONS else -1

        tier_map = {"free": 0, "pro": 1, "enterprise": 2}
        lang_map  = {"ko": 0, "en": 1, "mixed": 2}
        lang_mix  = ws.get("language_mix", {})

        feat = [
            meta.get("turn_index", 0),
            meta.get("elapsed_session_sec", 0),
            meta.get("budget_tokens_remaining", 0),
            ws.get("loc", 0),
            int(ws.get("git_dirty", False)),
            int(ws.get("last_ci_status", "none") == "failed"),
            int(ws.get("last_ci_status", "none") == "passed"),
            len(ws.get("open_files", [])),
            len(history),
            len(action_turns),
            last_action_idx,
            len(prompt),
            lang_mix.get("py", 0.0),
            lang_mix.get("js", 0.0),
            lang_mix.get("ts", 0.0),
            lang_mix.get("sql", 0.0),
            tier_map.get(meta.get("user_tier", "free"), 0),
            lang_map.get(meta.get("language_pref", "en"), 1),
            # 키워드 피처
            int(any(w in prompt for w in ["read", "open", "show", "view", "cat"])),
            int(any(w in prompt for w in ["write", "create", "new file"])),
            int(any(w in prompt for w in ["edit", "modify", "change", "update", "fix"])),
            int(any(w in prompt for w in ["grep", "search", "find", "pattern"])),
            int(any(w in prompt for w in ["run", "execute", "bash", "shell", "install"])),
            int(any(w in prompt for w in ["test", "pytest", "unittest"])),
            int(any(w in prompt for w in ["lint", "type", "mypy", "flake"])),
            int(any(w in prompt for w in ["web", "google", "documentation"])),
            int("?" in prompt or any(w in prompt for w in ["what", "how", "why", "which"])),
            int(any(w in prompt for w in ["plan", "step", "approach", "strategy"])),
            int(any(w in prompt for w in ["explain", "describe", "tell", "summarize"])),
            int(any(w in prompt for w in ["list", "directory", "folder", "ls"])),
        ]
        feat_list.append(feat)
    return np.array(feat_list, dtype=np.float32)


# ── LLM 추론 (GGUF) ──────────────────────────────────────────────────────────
def format_context(item: dict) -> str:
    """LLM용 컨텍스트 문자열 생성."""
    parts = []
    meta = item.get("session_meta", {})
    ws   = meta.get("workspace", {})

    parts.append(f"user_tier={meta.get('user_tier','unknown')}")
    parts.append(f"language={meta.get('language_pref','unknown')}")
    parts.append(f"turn={meta.get('turn_index',0)}")

    budget = meta.get("budget_tokens_remaining", 0)
    parts.append(f"budget={'LOW' if budget < 1000 else 'MID' if budget < 10000 else 'HIGH'}")
    parts.append(f"loc={ws.get('loc', 0)}")
    parts.append(f"ci={ws.get('last_ci_status', 'none')}")

    open_files = ws.get("open_files", [])
    if open_files:
        exts = {Path(fp).suffix.lstrip(".") for fp in open_files[:5] if Path(fp).suffix}
        if exts:
            parts.append(f"open_ext={','.join(sorted(exts))}")

    context = " | ".join(parts) + "\n"

    history_lines = []
    for turn in item.get("history", [])[-5:]:
        role = turn.get("role", "")
        if role == "user":
            history_lines.append(f"[user] {turn.get('content', '')[:200]}")
        elif role == "assistant_action":
            history_lines.append(
                f"[action:{turn.get('name','')}] {turn.get('result_summary','')[:100]}"
            )
    if history_lines:
        context += "\n".join(history_lines) + "\n"

    context += f"[current] {item.get('current_prompt', '')}"
    return context


def parse_action(output: str) -> Optional[str]:
    """LLM 출력 → action 레이블 파싱."""
    cleaned = output.strip().lower().split()[0] if output.strip() else ""
    if cleaned in ACTIONS:
        return cleaned
    for action in ACTIONS:
        if action in cleaned:
            return action
    kw_map = {
        "read": "read_file", "write": "write_file", "edit": "edit_file",
        "patch": "apply_patch", "grep": "grep_search", "glob": "glob_pattern",
        "list": "list_directory", "bash": "run_bash", "shell": "run_bash",
        "test": "run_tests", "lint": "lint_or_typecheck",
        "web": "web_search", "ask": "ask_user",
        "plan": "plan_task", "respond": "respond_only",
    }
    for kw, action in kw_map.items():
        if kw in cleaned:
            return action
    return None


def run_llm(llm, item: dict) -> Optional[str]:
    """단일 샘플 LLM 추론."""
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{format_context(item)}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    try:
        result = llm(
            prompt,
            max_tokens=16,
            temperature=0.0,
            stop=["<|im_end|>", "<|im_start|>"],
            echo=False,
        )
        raw = result["choices"][0]["text"]
        return parse_action(raw)
    except Exception as e:
        logger.debug(f"LLM 오류: {e}")
        return None


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(" Budget Router 추론 시작")
    logger.info("=" * 60)

    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    logger.info(f"테스트 데이터: {TEST_FILE}")
    items, ids = [], []
    with open(TEST_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                ids.append(obj["id"])
                items.append(obj)
    logger.info(f"  {len(items)}개 샘플")

    # ── LightGBM 로드 & 피처 추출 ────────────────────────────────────────────
    logger.info("LightGBM 모델 로드...")
    lgbm_models = load_lgbm_models(MODEL_DIR)

    logger.info("피처 추출 중...")
    features = extract_features(items, MODEL_DIR)

    logger.info("LightGBM 예측 중...")
    lgbm_preds, lgbm_probs_list, class_names = predict_lgbm(lgbm_models, features)

    t_lgbm = time.time()
    logger.info(f"  LightGBM 완료: {t_lgbm - t_start:.1f}초")

    import numpy as np
    max_probs = np.array([max(p) for p in lgbm_probs_list])
    logger.info(f"  평균 max_prob: {max_probs.mean():.3f}")

    # ── Budget Router ─────────────────────────────────────────────────────────
    uncertain_indices = list(np.where(max_probs < THRESHOLD)[0])

    # 최대 비율 제한 (가장 불확실한 것 우선)
    max_llm_count = int(len(items) * MAX_LLM_RATIO)
    if len(uncertain_indices) > max_llm_count:
        uncertain_indices = sorted(uncertain_indices, key=lambda i: max_probs[i])[:max_llm_count]

    logger.info(f"Budget Router: {len(uncertain_indices)}/{len(items)} → LLM "
                f"({len(uncertain_indices)/max(len(items),1)*100:.1f}%)")

    # ── LLM 로드 & 재예측 ─────────────────────────────────────────────────────
    final_preds = list(lgbm_preds)

    if uncertain_indices and GGUF_FILE.exists():
        elapsed = time.time() - t_start
        remaining = TIME_LIMIT - elapsed
        # T4 기준 ~30ms/sample 예상
        estimated_llm_sec = len(uncertain_indices) * 0.035
        logger.info(f"예상 LLM 소요: {estimated_llm_sec:.1f}s, 남은 시간: {remaining:.1f}s")

        if estimated_llm_sec < remaining * 0.85:
            try:
                from llama_cpp import Llama
                logger.info(f"GGUF 로드: {GGUF_FILE}")
                llm = Llama(
                    model_path=str(GGUF_FILE),
                    n_gpu_layers=-1,   # T4 GPU 전체 사용
                    n_ctx=2048,
                    n_threads=3,       # 3 vCPU
                    verbose=False,
                )

                replaced = 0
                for i in uncertain_indices:
                    pred = run_llm(llm, items[i])
                    if pred is not None:
                        final_preds[i] = pred
                        replaced += 1
                    # 시간 초과 방어
                    if time.time() - t_start > TIME_LIMIT:
                        logger.warning("시간 제한 도달 → LLM 중단")
                        break

                t_llm = time.time()
                logger.info(f"LLM 완료: {replaced}개 교체, {t_llm - t_lgbm:.1f}초")

            except ImportError:
                logger.warning("llama_cpp 없음 → LightGBM만 사용")
            except Exception as e:
                logger.warning(f"LLM 오류: {e} → LightGBM만 사용")
        else:
            logger.warning(f"시간 부족 → LLM 생략 (추정 {estimated_llm_sec:.1f}s > 허용 {remaining*0.85:.1f}s)")
    elif not GGUF_FILE.exists():
        logger.info("GGUF 없음 → LightGBM 단독 예측")

    # ── submission.csv 저장 ───────────────────────────────────────────────────
    logger.info(f"저장: {OUT_CSV}")
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "action"])
        for sid, pred in zip(ids, final_preds):
            writer.writerow([sid, pred])

    t_end = time.time()
    logger.info(f"완료! {len(ids)}개, {t_end - t_start:.1f}초")

    from collections import Counter
    dist = Counter(final_preds)
    logger.info(f"예측 분포: {dict(dist.most_common(5))}")


if __name__ == "__main__":
    main()
