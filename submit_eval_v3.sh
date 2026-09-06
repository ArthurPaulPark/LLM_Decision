#!/bin/bash
#SBATCH --job-name=QWEN_V3_EVAL         # Job 이름 지정
#SBATCH --partition=challenge_03        # guest03 전용 파티션
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                    # RTX 4500 Ada 1기 할당
#SBATCH --time=02:00:00                 # 최대 2시간 (평가는 빠름)
#SBATCH --output=qwen_v3_eval_%j.log    # 출력 결과는 여기에 저장됨

# 1. 작업 디렉토리 설정 및 환경 활성화
cd ${SLURM_SUBMIT_DIR}
echo "[INFO] Working Directory: ${SLURM_SUBMIT_DIR}"

source venv/bin/activate
echo "[INFO] Virtual Environment Activated."

# 2. 오프라인 모드 강제 세팅 (계산 노드 외부 인터넷 차단 대응)
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 3. V3 모델 상세 벤치마크 평가 스크립트 실행
# (V3는 V2 병합 모델 위에 새로운 LoRA가 올라간 구조이므로 베이스 모델을 병합본으로 지정합니다)
echo "[INFO] Qwen 1.5B V3 (Hard Sample) 모델 상세 평가 시작..."
python3 competition_agent_qwen15_v2/evaluate_v2.py \
    --model outputs/qwen15_v2/quantized/merged_bf16 \
    --adapter outputs/qwen15_v3_hard/best_model \
    --test-data data/processed/validation.json \
    --output outputs/qwen15_v3_hard/eval_detailed_results.json

echo "[INFO] V3 평가 태스크 완료!"
