#!/bin/bash
#SBATCH --job-name=QWEN_V3_HARD         # Job 이름 지정
#SBATCH --partition=challenge_03        # guest03 전용 파티션
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                    # RTX 4500 Ada 1기 할당
#SBATCH --time=12:00:00                 # 여유있게 12시간
#SBATCH --output=qwen_v3_hard_%j.log    # 출력 결과는 여기에 저장됨

set -e  # 오류 발생 시 즉시 중단

cd ${SLURM_SUBMIT_DIR}
echo "[INFO] Working Directory: ${SLURM_SUBMIT_DIR}"

source venv/bin/activate
echo "[INFO] Virtual Environment Activated."

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. 병합된 베이스 모델 준비 (V2 지식이 합쳐진 모델)
MERGED_DIR="outputs/qwen15_v2/quantized/merged_bf16"
if [ ! -d "$MERGED_DIR" ]; then
    echo "[INFO] V2 병합 모델이 없어서 생성을 시도합니다..."
    python3 quantize.py \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter outputs/qwen15_v2/best_model \
        --output outputs/qwen15_v2/quantized \
        --quant_type Q3_K_M
fi

# 2. Hard Sample 추출
mkdir -p data/processed_hard
HARD_TRAIN_FILE="data/processed_hard/train.json"
if [ ! -f "$HARD_TRAIN_FILE" ]; then
    echo "[INFO] Hard Sample 오답 데이터를 추출합니다..."
    python3 competition_agent_qwen15_v2/create_hard_samples.py \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --adapter outputs/qwen15_v2/best_model \
        --train_data data/processed/train.json \
        --output "$HARD_TRAIN_FILE" \
        --sample_size 15000 \
        --batch_size 16
else
    echo "[INFO] 이미 추출된 Hard Sample 데이터가 존재하여 스킵합니다: $HARD_TRAIN_FILE"
fi

# 3. Validation 데이터 세팅
echo "[INFO] 기존 Validation 데이터를 V3 디렉토리에 복사합니다..."
cp data/processed/validation.json data/processed_hard/validation.json

# 4. V3 Hard Sample 파인튜닝 시작
echo "[INFO] V3 Hard Sample 파인튜닝을 시작합니다..."
python3 competition_agent_qwen15_v2/train_v2.py \
    --config configs/qwen15_v3_hard.yaml

echo "[INFO] V3 태스크 완료!"
