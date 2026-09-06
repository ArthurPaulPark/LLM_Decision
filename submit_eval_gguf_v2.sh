#!/bin/bash
#SBATCH --job-name=QWEN_GGUF_EVAL
#SBATCH --partition=challenge_03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                    # llama.cpp의 GPU 가속(cuBLAS) 사용을 위해 1기 할당
#SBATCH --time=02:00:00
#SBATCH --output=qwen_gguf_eval_%j.log

# 1. 작업 디렉토리 설정 및 환경 활성화
cd ${SLURM_SUBMIT_DIR}
echo "[INFO] Working Directory: ${SLURM_SUBMIT_DIR}"

source venv/bin/activate
echo "[INFO] Virtual Environment Activated."

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 2. GGUF 모델 성능 검증 (Q4_K_M)
echo "[INFO] Q4_K_M GGUF 모델 벤치마크 평가를 시작합니다..."
python3 eval_gguf.py \
    --model outputs/qwen15_v2/quantized/model_q4km.gguf \
    --val_json data/processed/validation.json \
    --n_gpu_layers -1 \
    --n_ctx 2048

echo "[INFO] GGUF 벤치마크 태스크 완료!"
