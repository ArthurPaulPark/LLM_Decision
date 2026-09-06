#!/bin/bash
# llama.cpp 설치 & 빌드 스크립트
# 목적: quantize.py에서 필요한 llama-quantize, llama-imatrix 빌드
# 실행: 로그인 노드에서 최초 1회만 (인터넷 접속 필요)
#
# 사용법:
#   bash scripts/setup_llamacpp.sh
#   bash scripts/setup_llamacpp.sh --skip-clone   # 이미 클론됐으면 빌드만
#
# 결과물:
#   ./llama.cpp/build/bin/llama-quantize
#   ./llama.cpp/build/bin/llama-imatrix
#   ./llama.cpp/convert_hf_to_gguf.py (변환 스크립트)

set -euo pipefail

SKIP_CLONE=false
for arg in "$@"; do
    [[ "$arg" == "--skip-clone" ]] && SKIP_CLONE=true
done

LLAMACPP_DIR="$(pwd)/llama.cpp"
BUILD_DIR="${LLAMACPP_DIR}/build"

echo "============================================================"
echo " llama.cpp 빌드 (CPU-only, quantize 용도)"
echo " 출력 디렉토리: ${LLAMACPP_DIR}"
echo "============================================================"

# ── 1) 클론 ─────────────────────────────────────────────────────────────────
if [ "$SKIP_CLONE" = false ]; then
    if [ -d "$LLAMACPP_DIR" ]; then
        echo "[1/4] llama.cpp 이미 존재 → git pull 로 최신화..."
        cd "$LLAMACPP_DIR" && git pull --ff-only && cd ..
    else
        echo "[1/4] llama.cpp 클론 중 (shallow)..."
        git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$LLAMACPP_DIR"
    fi
else
    echo "[1/4] 클론 건너뜀 (--skip-clone)"
fi

# ── 2) Python 의존성 설치 ────────────────────────────────────────────────────
echo "[2/4] Python 의존성 설치 (gguf, sentencepiece, transformers)..."
# llama.cpp의 convert_hf_to_gguf.py 실행에 필요
pip install gguf sentencepiece --break-system-packages 2>/dev/null || \
pip install gguf sentencepiece 2>/dev/null || \
conda install -c conda-forge gguf sentencepiece -y 2>/dev/null || \
echo "      ⚠ pip/conda 설치 실패 - 수동 설치 필요: pip install gguf sentencepiece"

# ── 3) CMake 빌드 (CPU only) ─────────────────────────────────────────────────
echo "[3/4] CMake 빌드 중 (CPU-only 바이너리)..."
echo "      llama-quantize, llama-imatrix 대상"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_NATIVE=ON 2>&1 | tail -5

# 필요한 타겟만 빌드 (전체 빌드 방지)
make -j"$(nproc)" llama-quantize llama-imatrix 2>&1 | tail -10

cd -

# ── 4) 검증 ─────────────────────────────────────────────────────────────────
echo "[4/4] 빌드 결과 확인..."

OK=true
for bin in llama-quantize llama-imatrix; do
    path="${BUILD_DIR}/bin/${bin}"
    if [ -f "$path" ]; then
        size=$(du -sh "$path" | cut -f1)
        echo "      ✓ ${bin} (${size})"
    else
        echo "      ✗ ${bin} not found!"
        OK=false
    fi
done

convert_script="${LLAMACPP_DIR}/convert_hf_to_gguf.py"
if [ -f "$convert_script" ]; then
    echo "      ✓ convert_hf_to_gguf.py"
else
    echo "      ✗ convert_hf_to_gguf.py not found!"
    OK=false
fi

echo ""
if [ "$OK" = true ]; then
    echo "============================================================"
    echo " ✅ 빌드 완료! 이제 양자화 실행 가능:"
    echo ""
    echo "   python quantize.py \\"
    echo "     --adapter outputs/checkpoints \\"
    echo "     --output outputs/quantized"
    echo "============================================================"
else
    echo "============================================================"
    echo " ⚠ 일부 파일이 빌드되지 않았습니다."
    echo " 로그 확인 후 수동으로 빌드하세요:"
    echo "   cd llama.cpp/build && make llama-quantize llama-imatrix"
    echo "============================================================"
    exit 1
fi
