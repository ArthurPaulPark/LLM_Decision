#!/bin/bash
#SBATCH --job-name=QWEN_QUANTIZE        # Job 이름 지정
#SBATCH --partition=challenge_03        # guest03 전용 파티션
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8               # 병합/양자화 속도 향상을 위해 8코어 할당
#SBATCH --time=02:00:00                 # 최대 2시간 (병합 및 변환)
#SBATCH --output=qwen_quantize_%j.log   # 출력 결과는 여기에 저장됨

# 1. 작업 디렉토리 설정 및 환경 활성화
cd ${SLURM_SUBMIT_DIR}
echo "[INFO] Working Directory: ${SLURM_SUBMIT_DIR}"

source venv/bin/activate
echo "[INFO] Virtual Environment Activated."

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 2. 양자화 스크립트 실행 (IQ3_S)
echo "[INFO] 1.5B 모델과 V2 LoRA 가중치를 결합 후 IQ3_S으로 양자화를 시작합니다..."
python3 quantize.py \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter outputs/qwen15_v2/best_model \
    --output outputs/qwen15_v2/quantized \
    --quant_type IQ3_S

echo "[INFO] 양자화 태스크 완료!"
