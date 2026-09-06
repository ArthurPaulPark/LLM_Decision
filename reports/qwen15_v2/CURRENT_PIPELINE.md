# Current Pipeline Audit (Qwen 1.5B Production Baseline)

이 문서는 Qwen2.5-1.5B 프로덕션 최적화 캠페인을 수행하기 위한 기존 학습 파이프라인의 구조 및 설정을 요약합니다.

## 1. 하드웨어 및 기본 모델 환경
- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Data Type**: `bfloat16`
- **Device Map**: `auto`

## 2. 기존 학습 파라미터 (Training Settings)
- **YAML 파일**: `configs/competition_qlora.yaml`
- **Epochs**: 3
- **Batch Size**: 
  - Train: `16` (per-device)
  - Eval: `32` (per-device)
  - Gradient Accumulation Steps: `2`
  - 유효 배치 사이즈(Effective Batch Size): `32`
- **Learning Rate**: `2.0e-4`
- **Warmup Ratio**: `0.05`
- **Weight Decay**: `0.01`
- **Seed**: `42`
- **Mixed Precision**: `bf16`

## 3. 재사용 가능한 파이프라인 컴포넌트
- **학습 스크립트**: `train.py`
  - `ModelTrainer`와 `SFTTrainer` 통합 부분 재사용 가능.
  - Completion Only Loss (`DataCollatorForCompletionOnlyLM`) 구조 재사용 가능.
- **평가 및 벤치마크**: `evaluate.py`
  - Accuracy, Macro F1, Balanced Accuracy, Precision, Recall, Confusion Matrix 측정용 코드 재사용 가능.
- **병합 및 양자화**: `quantize.py`
  - `merge_lora` 및 llama.cpp 기반 GGUF F16 → IQ2_XXS/Q3_K_M 변환 로직 재사용 가능.

## 4. 적용할 핵심 수정 사항 (Optimization Elements)
1. **rsLoRA**: `LoraConfig`에 `use_rslora: true` 적용.
2. **Target Modules**: `gate_proj`, `up_proj`, `down_proj`를 명확히 추가 (기존 `q_proj`, `k_proj`, `v_proj`, `o_proj`에서 확대).
3. **Scheduler**: Cosine Scheduler (`lr_scheduler_type: "cosine"`), Warmup: 0.05.
4. **NEFTune**: `neftune_noise_alpha = 5`를 `SFTConfig`에 적용.
5. **LoRA+ 구현**: `SFTTrainer` 내부의 Optimizer Parameter Grouping을 통해 LoRA A (1e-4)와 B (4e-4)의 학습률을 각각 차등 분할 설정.
6. **검증 결과 데이터 저장**: 검증 시점마다 Predictions, Probabilities, Confidence, Confusion Matrix 등을 JSON/CSV 형태로 디스크에 영구 보관하여 추후 학습용 데이터로 재활용할 수 있게 가공.
7. **체크포인트 선택**: `Macro F1` 메트릭을 추적하여 최선의 가중치 선택 (`load_best_model_at_end = True`, `metric_for_best_model = "eval_macro_f1"`, `greater_is_better = True`).
8. **난수 시드 고정**: Python random, Numpy, Torch(CUDA 포함) 시드를 엄밀히 픽스.
9. **실시간 모니터링 로깅**: `Elapsed Time`, `Remaining Time`, `CPU/RAM`, `GPU` 지표를 실시간 출력하는 Callback 모듈 적용.
