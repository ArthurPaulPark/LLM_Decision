#!/bin/bash
#SBATCH --job-name=QWEN_V2_EVAL         # Job 이름 지정
#SBATCH --partition=challenge_03        # guest03 전용 파티션
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                    # RTX 4500 Ada 1기 할당
#SBATCH --time=02:00:00                 # 최대 2시간 (평가는 빠름)
#SBATCH --output=qwen_eval_%j.log       # 출력 결과는 여기에 저장됨

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
echo "[INFO] Hugging Face Offline Mode Enabled."

# 3. 평가 실행
echo "[INFO] Qwen 1.5B V2 모델 상세 평가 시작..."
python3 competition_agent_qwen15_v2/evaluate_v2.py \
    --config configs/qwen15_v2.yaml \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter outputs/qwen15_v2/best_model \
    --test-data ./data/processed/validation.json \
    --output ./outputs/qwen15_v2/eval_detailed_results.json

echo "[INFO] 평가 태스크 완료!"
