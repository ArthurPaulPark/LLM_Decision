#!/bin/bash
#SBATCH --job-name=QWEN15B_RANK32_EP3
#SBATCH --partition=challenge_03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=retrain_%j.log
#SBATCH --error=retrain_err_%j.log

# =============================================================================
# QLoRA 재학습 v2 (rank=32, epochs=3, completion-only loss)
#
# 변경사항:
#   - lora_rank: 16 → 32  (표현력 증가)
#   - epochs: 2 → 3
#   - completion-only loss (assistant 토큰만 loss 계산)
#
# 사용법:
#   sbatch submit_retrain.sh
#   squeue -u $USER   # 진행 확인
#   tail -f retrain_<JOBID>.log
# =============================================================================

set -e

echo "============================================================"
echo " QLoRA 재학습 v2: rank=32, epochs=3, completion-only loss"
echo " $(date)"
echo "============================================================"

cd ~/LLM_Decision

# conda 환경 활성화
echo "[1/6] conda 환경 활성화..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm_train

# GPU 상태 확인
echo "[2/6] GPU/CUDA 환경 확인..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    free, total = torch.cuda.mem_get_info(0)
    print(f'  VRAM: {total/1024**3:.1f}GB total, {free/1024**3:.1f}GB free')
"

# 이전 체크포인트 백업 (덮어쓰기 방지)
if [ -d "./outputs/checkpoints" ] && [ "$(ls -A ./outputs/checkpoints)" ]; then
    TS=$(date +%Y%m%d_%H%M%S)
    echo "[3/6] 이전 체크포인트 백업: outputs/checkpoints_backup_${TS}"
    mv ./outputs/checkpoints ./outputs/checkpoints_backup_${TS}
fi
mkdir -p ./outputs/checkpoints

# 데이터 전처리 (이미 있으면 스킵)
PROCESSED_DIR="./data/processed"
if [ ! -f "${PROCESSED_DIR}/train.json" ]; then
    echo "[4/6] 데이터 전처리..."
    python3 preprocess.py --config configs/competition_qlora.yaml
else
    echo "[4/6] 전처리 데이터 존재 → 스킵 (재전처리: rm -rf ${PROCESSED_DIR})"
fi

# 오프라인 모드 (compute 노드 인터넷 차단)
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# 모델 미리 다운로드 안 됐으면 로그인 노드에서:
# python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')"

# 학습 시작
echo "[5/6] 학습 시작..."
echo "      Config : configs/competition_qlora.yaml"
echo "      Model  : Qwen/Qwen2.5-1.5B-Instruct"
echo "      Method : QLoRA (4-bit NF4)"
echo "      Rank   : 32 (v1의 2배)"
echo "      Epochs : 3"
echo "      Loss   : Completion-only (assistant 토큰만)"
echo "      Batch  : 16 × 2 accum = 유효 32"

python3 train.py --config configs/competition_qlora.yaml

echo "[6/6] 학습 완료!"
echo ""
echo "  다음 단계 (로그인 노드에서 실행):"
echo "  1. 양자화:"
echo "     python quantize.py --adapter outputs/checkpoints --output outputs/quantized --quant_type Q3_K_M --skip_merge"
echo ""
echo "  2. 빠른 정확도 테스트:"
echo "     python gguf_inference.py --model outputs/quantized/model_q3km.gguf --test --n_test 50"
echo ""
echo "  3. 검증셋 전체 평가:"
echo "     python eval_gguf.py --model outputs/quantized/model_q3km.gguf"
echo ""
echo "============================================================"
echo " 완료: $(date)"
echo "============================================================"
