#!/bin/bash
#SBATCH --job-name=QWEN_V4_REHEARSAL
#SBATCH --partition=challenge_03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=qwen_v4_rehearsal_%j.log

set -e

cd ${SLURM_SUBMIT_DIR}
echo "[INFO] Working Directory: ${SLURM_SUBMIT_DIR}"

source venv/bin/activate
echo "[INFO] Virtual Environment Activated."

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. 데이터 배합 (Rehearsal Data 준비)
REHEARSAL_TRAIN="data/processed_v4_rehearsal/train.json"
if [ ! -f "$REHEARSAL_TRAIN" ]; then
    echo "[INFO] V4 리허설 데이터 배합을 시작합니다 (Easy 18k + Hard 4.4k)..."
    python3 scripts/prep_rehearsal_v4.py
else
    echo "[INFO] V4 리허설 데이터가 이미 존재합니다: $REHEARSAL_TRAIN"
fi

# Validation 데이터 복사
cp data/processed/validation.json data/processed_v4_rehearsal/validation.json

# 2. 파인튜닝
echo "[INFO] V4 Rehearsal 파인튜닝 시작..."
python3 competition_agent_qwen15_v2/train_v2.py \
    --config configs/qwen15_v4_rehearsal.yaml

# (train_v2.py가 자동으로 best_model을 outputs/qwen15_v4_rehearsal/best_model 에 저장함)

# 3. 양자화 (IQ3_S)
echo "[INFO] V4 모델 IQ3_S 양자화를 시작합니다..."
python3 quantize.py \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter outputs/qwen15_v4_rehearsal/best_model \
    --output outputs/qwen15_v4_rehearsal/quantized \
    --quant_type IQ3_S

# 4. FP16 어댑터 벤치마크 평가
echo "[INFO] V4 FP16 어댑터 벤치마크 평가..."
python3 competition_agent_qwen15_v2/evaluate_v2.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter outputs/qwen15_v4_rehearsal/best_model \
    --test-data data/processed/validation.json \
    --output outputs/qwen15_v4_rehearsal/eval_fp16_results.json

# 5. GGUF 양자화본 벤치마크 평가
echo "[INFO] V4 GGUF IQ3_S 단독 벤치마크 평가..."
python3 eval_gguf.py \
    --model outputs/qwen15_v4_rehearsal/quantized/model_iq3s.gguf \
    --val_json data/processed/validation.json \
    --n_gpu_layers -1 \
    --n_ctx 2048

echo "[INFO] V4 파이프라인 전체 완료!"
