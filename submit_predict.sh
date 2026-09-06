#!/bin/bash
#SBATCH --job-name=PREDICT
#SBATCH --partition=challenge_03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=predict_%j.log
#SBATCH --error=predict_error_%j.log

set -e

echo "============================================================"
echo " Competition Inference: Qwen3-4B + LoRA → submission.csv"
echo " $(date)"
echo "============================================================"

cd ~/LLM_Decision

# conda 환경 활성화
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm_train

# GPU 확인
python3 -c "
import torch
print(f'  GPU : {torch.cuda.get_device_name(0)}')
print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# 오프라인 모드 (compute 노드 인터넷 차단)
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 추론 실행
echo "[*] 추론 시작..."
python3 predict.py \
    --base_model Qwen/Qwen3-4B \
    --adapter outputs/checkpoints \
    --test_file data_alt/test.jsonl \
    --output submission.csv \
    --batch_size 8

echo "============================================================"
echo " ✓ 완료: submission.csv 생성됨"
echo " 완료 시각: $(date)"
echo "============================================================"
