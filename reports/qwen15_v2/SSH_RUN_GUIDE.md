# SSH Execution Guide (Campaign Phase 1)

이 가이드는 로컬에서 Push한 코드를 깃허브로부터 Pull 받아 원격 SSH 서버(Nvidia RTX 4500 Ada 24GB VRAM) 환경에서 최적화 파인튜닝 학습 및 벤치마크 평가를 연속적으로 구동하는 프로세스를 설명합니다.

## 🚀 전체 실행 워크플로우

### 1. SSH 원격 서버 접속 및 Git Pull
사용자의 로컬 코드가 깃허브에 push된 이후, 원격 서버에 접속하여 최신 소스코드를 pull 받습니다.
```bash
# 원격 서버 접속
ssh user@your-rtx4500-server-ip

# 작업 디렉터리로 이동 및 코드 동기화
cd ~/LLM_Decision
git pull origin main
```

---

### 2. 가상환경 및 의존성 확인
설치 요구사항을 확인하고 원격지 의존성 패키지를 점검합니다.
```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

### 3. Step 5: Smoke Test (max_steps = 10)
전체 훈련 수행 전, LoRA+, rsLoRA, NEFTune, Early Stopping 등의 기능들이 충돌 없이 원활히 연동되는지 검증하기 위해 10 step의 연기를 우선 시도합니다.
```bash
python competition_agent_qwen15_v2/train_v2.py \
    --config configs/qwen15_v2.yaml \
    --max_steps 10 \
    --rank 32
```
* **검증 내용**: 
  - 리소스 모니터링 로그 출력 정상 여부
  - LoRA+ 옵티마이저 파라미터 그룹 로딩 여부
  - 체크포인트 및 validation 데이터 저장 파일 (`outputs/qwen15_v2/validation/eval_results_step_10.json`) 생성 여부

---

### 4. Step 6: Full Training (Rank = 32)
Smoke Test가 통과하면 32 Rank 설정에 대해 전체 학습을 수행합니다.
```bash
# 학습 세션 끊김 방지용 tmux 기동 권장
tmux new -s qwen15_v2_train

# 전체 학습 실행
python competition_agent_qwen15_v2/train_v2.py \
    --config configs/qwen15_v2.yaml \
    --rank 32
```
* **모니터링**: 
  - 학습 실행 중 10 step 단위로 `Elapsed Time`, `Estimated Remaining Time`, `GPU/CPU/RAM Utilization`, `Loss` 등이 터미널에 실시간 갱신 출력됩니다.

---

### 5. Step 7: Merge & GGUF Quantization
학습 완료 후 베이스 모델과 LoRA 가중치를 BF16으로 병합한 뒤, 제출 및 프로덕션 호환을 위해 GGUF 양자화를 진행합니다.
```bash
# 병합 및 GGUF 변환 파이프라인 수행 (Q3_K_M 양자화)
python quantize.py \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter outputs/qwen15_v2/best_model \
    --output outputs/qwen15_v2/quantized \
    --quant_type Q3_K_M
```
* **출력물**: 
  - `outputs/qwen15_v2/quantized/model_q3km.gguf` 파일 (~840MB 내외)이 정상 생성되는지 확인합니다.

---

### 6. Step 8: Benchmark Evaluation & Report Generation
기존 프로덕션 성능과 V2 모델 성능을 교차 평가하고, 비교 보고서와 최종 보고서를 일괄 자동 작성합니다.

```bash
# 1. Baseline (기존 어댑터 적용 이전 혹은 base model) 평가 수행 및 기록
python competition_agent_qwen15_v2/evaluate_v2.py \
    --config configs/qwen15_v2.yaml \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --output outputs/qwen15_v2/evaluation/baseline_metrics.json

# 2. 신규 V2 최적화 모델 평가 수행 및 기록
python competition_agent_qwen15_v2/evaluate_v2.py \
    --config configs/qwen15_v2.yaml \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter outputs/qwen15_v2/best_model \
    --output outputs/qwen15_v2/evaluation/v2_metrics.json

# 3. COMPARISON.md 및 FINAL_REPORT.md 자동 생성
python competition_agent_qwen15_v2/compare_v2.py \
    --base_json outputs/qwen15_v2/evaluation/baseline_metrics.json \
    --v2_json outputs/qwen15_v2/evaluation/v2_metrics.json \
    --out_dir reports/qwen15_v2 \
    --rank 32
```
* **결과 확인**:
  - [reports/qwen15_v2/COMPARISON.md](file://~/LLM_Decision/reports/qwen15_v2/COMPARISON.md) 및 [reports/qwen15_v2/FINAL_REPORT.md](file://~/LLM_Decision/reports/qwen15_v2/FINAL_REPORT.md)가 정량 수치 기반으로 바르게 저장되었는지 열어 확인합니다.
  - 리포트 최하단의 **ADOPT** 또는 **REJECT** 표기 결과를 관찰합니다.
