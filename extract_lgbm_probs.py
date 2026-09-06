#!/usr/bin/env python3
"""
submit.zip 패키지에서 LightGBM 14클래스 확률 CSV 추출.

기존 script.py는 예측 라벨만 출력하지만, budget_router는 확률이 필요함.
이 스크립트는 submit 패키지를 직접 임포트해서 확률 포함 CSV를 생성.

사용법:
  # submit.zip 압축 해제
  mkdir -p ~/submit_extracted && cd ~/submit_extracted
  unzip ~/LLM_Decision/submit.zip

  # 확률 CSV 생성
  cd ~/LLM_Decision
  python extract_lgbm_probs.py \\
      --submit_dir ~/submit_extracted \\
      --test_file data_alt/test.jsonl \\
      --output lgbm_probs.csv

  # val 데이터도 생성 (budget router 튜닝용)
  python extract_lgbm_probs.py \\
      --submit_dir ~/submit_extracted \\
      --test_file data_alt/train.jsonl \\
      --labels_file data_alt/train_labels.csv \\
      --output lgbm_val_probs.csv \\
      --max_samples 5000
"""

import sys
import json
import csv
import logging
import argparse
import numpy as np
from pathlib import Path

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


def load_submit_package(submit_dir: str):
    """
    submit 패키지에서 LightGBM 모델과 feature extractor 로드.

    submit.zip 구조 (예상):
      model/lgbm_fold1.txt ~ fold5.txt
      model/lib/feature_extractor_v2.py
      auto_features.py
      model/svd_prompt_components.npy
      model/svd_history_components.npy
      model/feature_schema.json
      model/group_stats.json
    """
    submit_dir = Path(submit_dir)

    # 패키지 경로를 sys.path에 추가
    sys.path.insert(0, str(submit_dir))
    sys.path.insert(0, str(submit_dir / "model" / "lib"))

    import lightgbm as lgb

    # LightGBM 모델 로드 (5-fold 앙상블)
    model_dir = submit_dir / "model"
    models = []
    for i in range(1, 6):
        model_path = model_dir / f"lgbm_fold{i}.txt"
        if model_path.exists():
            booster = lgb.Booster(model_file=str(model_path))
            models.append(booster)
            logger.info(f"✓ LightGBM fold{i} 로드")

    if not models:
        raise FileNotFoundError(f"{model_dir}에 lgbm_fold*.txt 없음")

    logger.info(f"✓ 총 {len(models)}개 모델 로드")
    return models, submit_dir


def extract_features(submit_dir: Path, items: list):
    """
    submit 패키지의 feature_extractor_v2 사용해서 특징 추출.
    패키지 구조에 따라 조정 필요.
    """
    try:
        # feature_extractor_v2.py 임포트 시도
        from feature_extractor_v2 import FeatureExtractor
        logger.info("✓ feature_extractor_v2 임포트 성공")

        # 필요한 모델 파일 로드
        model_dir = submit_dir / "model"
        extractor = FeatureExtractor(str(model_dir))
        features = extractor.transform(items)
        return features

    except ImportError:
        logger.warning("feature_extractor_v2 임포트 실패. auto_features.py 시도...")

    try:
        # auto_features.py 임포트 시도
        import auto_features
        features = auto_features.extract(items, str(submit_dir / "model"))
        return features

    except (ImportError, AttributeError):
        logger.warning("auto_features.py 실패. script.py 직접 분석 필요.")

    # 마지막 수단: script.py를 동적으로 실행하는 방식
    # (패키지 구조가 다를 경우 이 부분을 수정)
    raise RuntimeError(
        "feature_extractor 임포트 실패.\n"
        "submit.zip 내부 구조를 확인하고 이 함수를 수정하세요.\n"
        "또는 script.py를 직접 수정해 probability를 출력하게 하세요."
    )


def predict_with_probs(models, features) -> tuple:
    """
    LightGBM 앙상블로 14클래스 확률 예측.

    Returns:
        probs: (N, 14) 평균 확률
        preds: 예측 라벨 리스트
    """
    # 각 fold 예측 → 평균
    fold_probs = []
    for model in models:
        prob = model.predict(features)  # (N, 14)
        fold_probs.append(prob)

    probs = np.mean(fold_probs, axis=0)  # (N, 14)

    # 클래스 이름 (LightGBM 학습 시 사용한 순서와 일치해야 함)
    # 모델에서 직접 클래스 목록 가져오기 시도
    try:
        class_names = models[0].class_names()  # LightGBM ≥ 4.0
    except Exception:
        class_names = ACTIONS  # fallback

    # 예측 라벨
    pred_indices = probs.argmax(axis=1)
    preds = [class_names[i] if i < len(class_names) else "respond_only" for i in pred_indices]

    return probs, preds, class_names


def main():
    parser = argparse.ArgumentParser(description="LightGBM 확률 CSV 추출")
    parser.add_argument("--submit_dir", required=True,
                        help="압축 해제된 submit 패키지 디렉토리")
    parser.add_argument("--test_file", default="data_alt/test.jsonl",
                        help="예측할 JSONL 파일")
    parser.add_argument("--output", default="lgbm_probs.csv",
                        help="출력 CSV (id, pred_action, class1, ..., class14)")
    parser.add_argument("--labels_file", default=None,
                        help="정답 레이블 CSV (정확도 평가용, 선택)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="샘플 수 제한 (테스트용)")
    args = parser.parse_args()

    # 데이터 로드
    logger.info(f"데이터 로드: {args.test_file}")
    items = []
    with open(args.test_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if args.max_samples:
        items = items[:args.max_samples]
    ids = [item["id"] for item in items]
    logger.info(f"  {len(items)}개 샘플")

    # submit 패키지 로드
    models, submit_dir = load_submit_package(args.submit_dir)

    # 특징 추출
    logger.info("특징 추출 중...")
    features = extract_features(submit_dir, items)
    logger.info(f"  feature shape: {features.shape}")

    # 확률 예측
    logger.info("LightGBM 예측 중...")
    probs, preds, class_names = predict_with_probs(models, features)
    logger.info(f"  probs shape: {probs.shape}, classes: {class_names[:3]}...")

    # CSV 저장 (id, pred_action, class1, ..., class14)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "pred_action"] + class_names)
        for i, (sid, pred) in enumerate(zip(ids, preds)):
            row = [sid, pred] + [f"{p:.6f}" for p in probs[i]]
            writer.writerow(row)

    logger.info(f"✓ 저장: {args.output} ({len(ids)}개)")

    # 정확도 평가 (레이블 있을 경우)
    if args.labels_file:
        labels = {}
        with open(args.labels_file, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                labels[row["id"]] = row["action"]

        correct = sum(1 for sid, pred in zip(ids, preds) if labels.get(sid) == pred)
        acc = correct / len(ids) if ids else 0
        logger.info(f"\nLightGBM 정확도: {correct}/{len(ids)} = {acc*100:.2f}%")

        # 확신도 분포
        max_probs = probs.max(axis=1)
        logger.info(f"평균 max_prob: {max_probs.mean():.3f}")
        for thresh in [0.5, 0.6, 0.65, 0.7, 0.8]:
            n_low = (max_probs < thresh).sum()
            logger.info(f"  max_prob < {thresh}: {n_low:,} ({n_low/len(ids)*100:.1f}%)")

    logger.info(f"\n다음 단계:")
    logger.info(f"  python budget_router.py \\")
    logger.info(f"      --lgbm_probs {args.output} \\")
    logger.info(f"      --test_file {args.test_file} \\")
    logger.info(f"      --gguf_model outputs/quantized/model_q3km.gguf \\")
    logger.info(f"      --threshold 0.65 \\")
    logger.info(f"      --output submission_hybrid.csv")


if __name__ == "__main__":
    main()
