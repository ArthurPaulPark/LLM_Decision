#!/bin/bash
# =============================================================================
# submit.zip 빌드 스크립트
#
# 최종 구조:
#   submit.zip
#   ├── model/
#   │   ├── lgbm_fold1.txt ~ lgbm_fold5.txt  ← LightGBM 5-fold
#   │   ├── lib/
#   │   │   └── feature_extractor_v2.py       ← 피처 추출기
#   │   ├── svd_prompt_components.npy          ← SVD (있을 경우)
#   │   ├── svd_history_components.npy
#   │   ├── feature_schema.json
#   │   ├── group_stats.json
#   │   └── model.gguf                         ← GGUF (Budget Router)
#   ├── script.py
#   └── requirements.txt
#
# 사용법:
#   bash scripts/build_submit.sh                     # Q3_K_M (기본)
#   bash scripts/build_submit.sh --gguf Q3_K_S       # Q3_K_S 사용
#   bash scripts/build_submit.sh --lgbm_only         # GGUF 없이 LightGBM만
#
# 기존 submit 패키지 위치:
#   ~/submit_extracted  (또는 SUBMIT_EXTRACTED 환경변수로 지정)
# =============================================================================

set -euo pipefail
cd ~/LLM_Decision

# ── 설정 ────────────────────────────────────────────────────────────────────
SUBMIT_EXTRACTED="${SUBMIT_EXTRACTED:-${HOME}/submit_extracted}"
QUANT_TYPE="Q3_K_M"
LGBM_ONLY=false

for arg in "$@"; do
    case $arg in
        --gguf=*)   QUANT_TYPE="${arg#*=}" ;;
        --lgbm_only) LGBM_ONLY=true ;;
    esac
done

# GGUF 파일 경로
QUANT_SUFFIX=$(echo "$QUANT_TYPE" | tr '[:upper:]' '[:lower:]' | tr -d '_')
GGUF_FILE="outputs/quantized/model_${QUANT_SUFFIX}.gguf"

echo "============================================================"
echo " submit.zip 빌드 ($(date))"
echo " GGUF: $( [ "$LGBM_ONLY" = true ] && echo '없음 (LightGBM only)' || echo ${GGUF_FILE} )"
echo " 기존 submit 패키지: ${SUBMIT_EXTRACTED}"
echo "============================================================"

# ── 파일 확인 ───────────────────────────────────────────────────────────────
echo ""
echo "[1/5] 파일 확인..."

if [ ! -d "${SUBMIT_EXTRACTED}" ]; then
    echo "✗ ${SUBMIT_EXTRACTED} 없음!"
    echo "  mkdir -p ~/submit_extracted && unzip ~/LLM_Decision/submit.zip -d ~/submit_extracted"
    exit 1
fi

# LightGBM 모델 확인
FOLD_COUNT=$(ls "${SUBMIT_EXTRACTED}/model/"lgbm_fold*.txt 2>/dev/null | wc -l)
if [ "${FOLD_COUNT}" -eq 0 ]; then
    echo "✗ lgbm_fold*.txt 없음 (경로: ${SUBMIT_EXTRACTED}/model/)"
    exit 1
fi
echo "  ✓ LightGBM ${FOLD_COUNT}개 fold 발견"

# GGUF 확인
if [ "$LGBM_ONLY" = false ]; then
    if [ ! -f "${GGUF_FILE}" ]; then
        echo "✗ GGUF 없음: ${GGUF_FILE}"
        echo "  python quantize.py --quant_type ${QUANT_TYPE}"
        exit 1
    fi
    GGUF_MB=$(du -sm "${GGUF_FILE}" | cut -f1)
    echo "  ✓ GGUF: ${GGUF_MB}MB"
fi

# ── 빌드 디렉토리 준비 ──────────────────────────────────────────────────────
echo ""
echo "[2/5] 빌드 디렉토리 준비..."
BUILD_DIR="submit_build"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}/model/lib"

# ── 파일 복사 ───────────────────────────────────────────────────────────────
echo ""
echo "[3/5] 파일 복사..."

# script.py, requirements.txt
cp script.py            "${BUILD_DIR}/script.py"
cp requirements_submit.txt "${BUILD_DIR}/requirements.txt"
echo "  ✓ script.py, requirements.txt"

# LightGBM 모델 파일 복사
cp "${SUBMIT_EXTRACTED}/model/"lgbm_fold*.txt "${BUILD_DIR}/model/"
echo "  ✓ lgbm_fold*.txt (${FOLD_COUNT}개)"

# feature_extractor_v2.py (model/lib/)
if [ -d "${SUBMIT_EXTRACTED}/model/lib" ]; then
    cp -r "${SUBMIT_EXTRACTED}/model/lib/." "${BUILD_DIR}/model/lib/"
    echo "  ✓ model/lib/ ($(ls "${BUILD_DIR}/model/lib/" | wc -l)개 파일)"
else
    echo "  ⚠ model/lib/ 없음 (feature_extractor_v2 없이 fallback 사용됨)"
fi

# SVD 컴포넌트 복사 (있을 경우)
for npy_file in "${SUBMIT_EXTRACTED}/model/"*.npy; do
    [ -f "$npy_file" ] && cp "$npy_file" "${BUILD_DIR}/model/" && \
        echo "  ✓ $(basename $npy_file)"
done || true

# JSON 파일 복사 (feature_schema, group_stats 등)
for json_file in "${SUBMIT_EXTRACTED}/model/"*.json; do
    [ -f "$json_file" ] && cp "$json_file" "${BUILD_DIR}/model/" && \
        echo "  ✓ $(basename $json_file)"
done || true

# 기타 .pkl 파일 복사
for pkl_file in "${SUBMIT_EXTRACTED}/model/"*.pkl; do
    [ -f "$pkl_file" ] && cp "$pkl_file" "${BUILD_DIR}/model/" && \
        echo "  ✓ $(basename $pkl_file)"
done || true

# auto_features.py (최상위에 있을 경우)
if [ -f "${SUBMIT_EXTRACTED}/auto_features.py" ]; then
    cp "${SUBMIT_EXTRACTED}/auto_features.py" "${BUILD_DIR}/"
    echo "  ✓ auto_features.py"
fi

# GGUF 모델 복사
if [ "$LGBM_ONLY" = false ]; then
    cp "${GGUF_FILE}" "${BUILD_DIR}/model/model.gguf"
    echo "  ✓ model.gguf (${GGUF_MB}MB)"
fi

# ── 크기 확인 ───────────────────────────────────────────────────────────────
echo ""
echo "[4/5] 크기 확인..."
BUILD_MB=$(du -sm "${BUILD_DIR}" | cut -f1)
echo "  빌드 디렉토리: ${BUILD_MB}MB"

if [ "${BUILD_MB}" -gt 1000 ]; then
    echo "  ❌ ${BUILD_MB}MB > 1GB! 더 작은 양자화 필요"
    echo "     bash scripts/build_submit.sh --gguf Q3_K_S"
    echo "     bash scripts/build_submit.sh --lgbm_only"
    exit 1
else
    echo "  ✅ ${BUILD_MB}MB < 1GB"
fi

# ── zip 생성 ────────────────────────────────────────────────────────────────
echo ""
echo "[5/5] zip 생성..."
OUTPUT_ZIP="submit_$(date +%Y%m%d_%H%M%S).zip"
cd "${BUILD_DIR}"
zip -r "../${OUTPUT_ZIP}" . -x "*.DS_Store" -x "__pycache__/*" -x "*.pyc"
cd ..

ZIP_MB=$(du -sm "${OUTPUT_ZIP}" | cut -f1)
echo "  ✓ ${OUTPUT_ZIP} (${ZIP_MB}MB)"

echo ""
echo "구조 확인:"
unzip -l "${OUTPUT_ZIP}" | grep -E "(model/|script|requirements)" | head -20

echo ""
echo "============================================================"
if [ "${ZIP_MB}" -le 1024 ]; then
    echo " ✅ 빌드 완료! ${OUTPUT_ZIP} (${ZIP_MB}MB)"
else
    echo " ❌ ${ZIP_MB}MB > 1GB"
fi
echo "============================================================"
echo ""
echo "Mac으로 다운로드:"
echo "  scp <서버주소>:~/LLM_Decision/${OUTPUT_ZIP} ~/Downloads/"
echo ""
echo "DACON 제출:"
echo "  https://dacon.io/competitions/official/236694/mysubmission"
