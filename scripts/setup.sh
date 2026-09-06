#!/bin/bash
# SSH 서버 최초 환경 세팅 스크립트
# RTX 4500 Ada (24GB) + Qwen2.5-7B-Instruct + QLoRA
# 사용법: cd ~/LLM_Decision && bash scripts/setup.sh

set -e

echo "============================================================"
echo " LLM Decision - 환경 세팅 (RTX 4500 Ada)"
echo "============================================================"

cd ~/LLM_Decision

# Python 버전 확인
PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✓ Python: $PYTHON_VER"

# 가상환경 생성
if [ ! -d "venv" ]; then
    echo "📦 가상환경 생성..."
    python3 -m venv venv
fi
source venv/bin/activate
echo "✓ 가상환경 활성화"

# pip 업그레이드
pip install --upgrade pip -q

# CUDA 버전에 맞는 PyTorch 설치 (CUDA 12.1 기준)
# 서버 CUDA 버전 확인 후 필요시 cu118 등으로 변경
echo "📦 PyTorch (CUDA 12.1) 설치..."
pip install torch --index-url https://download.pytorch.org/whl/cu124 -q
# torchvision은 텍스트 LLM 파인튜닝에 불필요하지만 transformers 내부 의존성으로 필요시 추가
# pip install torchvision --index-url https://download.pytorch.org/whl/cu124 -q

# GPU 확인 (로그인/스토리지 노드는 GPU 없음 → 경고만 출력, 정상)
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
if torch.cuda.is_available():
    print(f'  GPU : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'  CUDA: {torch.version.cuda}')
else:
    print('  [INFO] CUDA 없음 - 로그인 노드에서는 정상. sbatch 실행 시 GPU 노드에서 동작합니다.')
"

# 나머지 패키지 설치
echo "📦 파인튜닝 패키지 설치..."
pip install -r requirements.txt -q

# 디렉토리 생성
echo "📂 디렉토리 생성..."
mkdir -p data/{raw,processed,cache}
mkdir -p models
mkdir -p outputs/{checkpoints,adapters,predictions,tensorboard}
mkdir -p logs

echo ""
echo "============================================================"
echo " [선택] Qwen2.5-7B-Instruct 사전 다운로드 (온라인 상태에서 실행)"
echo "   python3 -c \"from huggingface_hub import snapshot_download;"
echo "   snapshot_download('Qwen/Qwen2.5-7B-Instruct')\""
echo "============================================================"
echo ""
echo "============================================================"
echo " ✓ 환경 세팅 완료!"
echo ""
echo " 다음 단계:"
echo "   1. 데이터 전처리:  python preprocess.py --config configs/rtx4500_qlora.yaml"
echo "   2. 학습 제출:      sbatch submit_train.sh"
echo "   3. 로그 확인:      tail -f qwen_train_<JOB_ID>.log"
echo "   4. TensorBoard:   tensorboard --logdir outputs/tensorboard --port 6006"
echo "============================================================"
