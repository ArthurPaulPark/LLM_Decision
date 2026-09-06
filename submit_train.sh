#!/bin/bash
#SBATCH --job-name=QWEN25_1B5_QLORA
#SBATCH --partition=challenge_03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=qwen_train_%j.log
#SBATCH --error=qwen_error_%j.log

set -e

echo "============================================================"
echo " QLoRA Fine-tuning: Qwen2.5-1.5B-Instruct @ RTX 4500 Ada"
echo " $(date)"
echo "============================================================"

cd ~/LLM_Decision

# conda 환경 활성화
echo "[1/5] conda 환경 활성화 (llm_train / Python 3.11)..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm_train

# GPU 상태 확인
echo "[2/5] GPU/CUDA 환경 확인..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  Device: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'  CUDA Version: {torch.version.cuda}')
"

# 데이터 전처리
PROCESSED_DIR="./data/processed"
if [ ! -f "${PROCESSED_DIR}/train.json" ]; then
    echo "[3/5] 데이터 전처리 시작..."
    python3 preprocess.py --config configs/competition_qlora.yaml
else
    echo "[3/5] 전처리 데이터 이미 존재 → 건너뜀"
    echo "      (재전처리: rm -rf ${PROCESSED_DIR} 후 재실행)"
fi

# compute 노드는 인터넷 차단 → 오프라인 모드 강제
# 모델 사전 다운로드 (로그인 노드에서):
#   python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 학습 시작
echo "[4/5] QLoRA 학습 시작..."
echo "      Config: configs/competition_qlora.yaml"
echo "      Model : Qwen/Qwen2.5-1.5B-Instruct"
echo "      Method: QLoRA (4-bit NF4 + rank=16)"
echo "      Batch : 16 (유효 배치 32)"
echo "      Epoch : 2"
echo "      목표  : GGUF IQ2_XXS 양자화 → ~620MB"

python3 train.py --config configs/competition_qlora.yaml

echo "[5/5] 학습 완료!"
echo "      체크포인트: ./outputs/checkpoints/"
echo "      다음 단계:  python quantize.py  (로그인 노드에서)"
echo "============================================================"
echo " 완료 시각: $(date)"
echo "============================================================"
