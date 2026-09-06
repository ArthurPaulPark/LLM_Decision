#!/bin/bash
# =============================================================================
# 전체 파이프라인 실행 스크립트 (로그인 노드에서 실행)
#
# 단계:
#   A) SLURM 재학습 제출 → 완료 대기
#   B) Q3_K_M 양자화
#   C) GGUF 빠른 정확도 테스트 (50샘플)
#   D) LightGBM 확률 CSV 추출 (submit 패키지)
#   E) Budget Router 실행 → submission_hybrid.csv
#   F) 제출 zip 빌드
#
# 사용법:
#   # A부터 전체:
#   bash scripts/run_pipeline.sh
#
#   # 재학습 건너뛰고 B부터:
#   bash scripts/run_pipeline.sh --skip_train
#
#   # 양자화도 있으면 C부터:
#   bash scripts/run_pipeline.sh --skip_train --skip_quant
# =============================================================================

set -euo pipefail

# ── 설정 ────────────────────────────────────────────────────────────────────
SUBMIT_DIR="${HOME}/submit_extracted"
SUBMIT_ZIP="${HOME}/LLM_Decision/submit.zip"   # 현재 제출 zip 위치 (필요 시 수정)
GGUF_MODEL="outputs/quantized/model_q3km.gguf"
LGBM_PROBS_CSV="lgbm_probs.csv"
HYBRID_CSV="submission_hybrid.csv"
THRESHOLD=0.65

SKIP_TRAIN=false
SKIP_QUANT=false

for arg in "$@"; do
    [[ "$arg" == "--skip_train" ]] && SKIP_TRAIN=true
    [[ "$arg" == "--skip_quant" ]] && SKIP_QUANT=true
done

cd ~/LLM_Decision
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm_train

echo "============================================================"
echo " LLM_Decision 전체 파이프라인"
echo " $(date)"
echo "============================================================"

# ── A) SLURM 재학습 ─────────────────────────────────────────────────────────
if [ "$SKIP_TRAIN" = false ]; then
    echo ""
    echo "[A] SLURM 재학습 제출 (rank=32, epochs=3, completion-only loss)"
    JOB_ID=$(sbatch submit_retrain.sh | awk '{print $NF}')
    echo "    Job ID: ${JOB_ID}"
    echo "    로그  : tail -f retrain_${JOB_ID}.log"
    echo ""
    echo "    학습 완료까지 대기 중..."

    # 학습 완료 대기 (10분 간격 체크)
    while true; do
        STATE=$(squeue -j "${JOB_ID}" -h -o "%T" 2>/dev/null || echo "DONE")
        if [[ "$STATE" == "DONE" || "$STATE" == "" ]]; then
            echo "    ✓ 학습 완료!"
            break
        fi
        echo "    현재 상태: ${STATE} | $(date '+%H:%M:%S')"
        sleep 600  # 10분 대기
    done

    # 체크포인트 확인
    if [ ! -d "outputs/checkpoints" ] || [ -z "$(ls outputs/checkpoints/)" ]; then
        echo "    ✗ 체크포인트 없음! 로그 확인: retrain_${JOB_ID}.log"
        exit 1
    fi
    echo "    체크포인트: $(ls outputs/checkpoints/ | tail -1)"
else
    echo "[A] 학습 건너뜀 (--skip_train)"
fi

# ── B) Q3_K_M 양자화 ────────────────────────────────────────────────────────
if [ "$SKIP_QUANT" = false ]; then
    echo ""
    echo "[B] Q3_K_M 양자화..."

    # llama.cpp 빌드 확인
    if [ ! -f "llama.cpp/build/bin/llama-quantize" ]; then
        echo "    llama-quantize 없음 → setup_llamacpp.sh 실행"
        bash scripts/setup_llamacpp.sh
    fi

    if [ -f "${GGUF_MODEL}" ]; then
        echo "    기존 GGUF 백업..."
        mv "${GGUF_MODEL}" "${GGUF_MODEL}.bak_$(date +%Y%m%d_%H%M%S)"
    fi

    python quantize.py \
        --adapter outputs/checkpoints \
        --output outputs/quantized \
        --quant_type Q3_K_M

    echo "    ✓ 양자화 완료: ${GGUF_MODEL}"
    ls -lh "${GGUF_MODEL}"
else
    echo "[B] 양자화 건너뜀 (--skip_quant)"
    if [ ! -f "${GGUF_MODEL}" ]; then
        echo "    ✗ ${GGUF_MODEL} 없음! --skip_quant 제거 후 재실행"
        exit 1
    fi
fi

# ── C) GGUF 빠른 정확도 테스트 ──────────────────────────────────────────────
echo ""
echo "[C] GGUF 정확도 테스트 (훈련 데이터 50샘플)..."
python gguf_inference.py \
    --model "${GGUF_MODEL}" \
    --test \
    --n_test 50

echo ""
echo "  ↑ 65% 미만이면 재학습이 더 필요함."
echo "  계속 진행할까요? (Enter=계속, Ctrl+C=중단)"
read -r

# ── D) LightGBM 확률 CSV 추출 ────────────────────────────────────────────────
echo ""
echo "[D] LightGBM 확률 CSV 추출..."

if [ ! -d "${SUBMIT_DIR}" ]; then
    echo "    submit_extracted 없음 → submit.zip 압축 해제"
    if [ ! -f "${SUBMIT_ZIP}" ]; then
        echo "    ✗ ${SUBMIT_ZIP} 없음!"
        echo "    SUBMIT_ZIP 변수를 올바른 경로로 수정하세요."
        exit 1
    fi
    mkdir -p "${SUBMIT_DIR}"
    unzip "${SUBMIT_ZIP}" -d "${SUBMIT_DIR}"
fi

python extract_lgbm_probs.py \
    --submit_dir "${SUBMIT_DIR}" \
    --test_file data_alt/test.jsonl \
    --output "${LGBM_PROBS_CSV}"

echo "    ✓ LightGBM 확률 저장: ${LGBM_PROBS_CSV}"

# 검증셋 확률도 추출 (threshold 튜닝용)
echo "    검증셋 확률 추출..."
python extract_lgbm_probs.py \
    --submit_dir "${SUBMIT_DIR}" \
    --test_file data_alt/train.jsonl \
    --labels_file data_alt/train_labels.csv \
    --output lgbm_train_probs.csv \
    --max_samples 5000 || echo "    (검증 추출 실패 - 계속 진행)"

# ── E) Budget Router 실행 ────────────────────────────────────────────────────
echo ""
echo "[E] Budget Router 실행 (threshold=${THRESHOLD})..."

# 먼저 dry_run으로 라우팅 분석
python budget_router.py \
    --lgbm_probs "${LGBM_PROBS_CSV}" \
    --threshold "${THRESHOLD}" \
    --output /dev/null \
    --dry_run

echo ""
echo "  위 분석 결과 확인. LLM 비율 15-20%이면 계속 진행."
echo "  계속할까요? (Enter=계속, Ctrl+C=중단)"
read -r

# 실제 Budget Router 실행
python budget_router.py \
    --lgbm_probs "${LGBM_PROBS_CSV}" \
    --test_file data_alt/test.jsonl \
    --gguf_model "${GGUF_MODEL}" \
    --threshold "${THRESHOLD}" \
    --max_llm_ratio 0.18 \
    --n_gpu_layers -1 \
    --output "${HYBRID_CSV}"

echo "    ✓ Hybrid 예측 저장: ${HYBRID_CSV}"
wc -l "${HYBRID_CSV}"

# ── F) 제출 zip 빌드 ────────────────────────────────────────────────────────
echo ""
echo "[F] 제출 파일 준비..."
echo "    submission_hybrid.csv 를 대회 제출 포맷으로 변환이 필요한지 확인하세요."
echo "    현재 포맷: id, action"
head -5 "${HYBRID_CSV}"
echo ""
echo "    ※ 현재 제출 방식이 zip이라면:"
echo "       cp ${HYBRID_CSV} <submit_dir>/predictions.csv"
echo "       cd <submit_dir> && zip -r new_submission.zip ."
echo ""
echo "    ※ 현재 제출 방식이 CSV 직접이라면:"
echo "       ${HYBRID_CSV} 파일을 그대로 제출하세요."

echo ""
echo "============================================================"
echo " 파이프라인 완료: $(date)"
echo "============================================================"
echo ""
echo " 파일 요약:"
echo "   GGUF 모델    : $(du -sh ${GGUF_MODEL} 2>/dev/null || echo '없음')"
echo "   LightGBM 확률: $(wc -l < ${LGBM_PROBS_CSV} 2>/dev/null || echo '없음') 행"
echo "   최종 예측    : $(wc -l < ${HYBRID_CSV} 2>/dev/null || echo '없음') 행"
