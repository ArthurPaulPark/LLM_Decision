# Current Pipeline Analysis (Teacher Baseline)

이 문서는 AI 의사결정 대회용 Teacher 모델 학습 파이프라인의 기존 설정 분석 결과를 정리합니다.

## 1. 하드웨어 및 기본 모델 환경
- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Data Type**: `bfloat16`
- **Device Map**: `auto` (GPU/CPU 가속 및 병렬 배정)

## 2. 학습 파라미터 및 프레임워크 기능 (Training Arguments)
- **Epochs**: 3
- **Batch Size**: 
  - Train: `16` (per-device)
  - Eval: `32` (per-device)
  - Gradient Accumulation Steps: `2`
  - 유효 배치 사이즈(Effective Batch Size): `32` (Train 기준, `16 * 2`)
- **Learning Rate**: `2.0e-4`
- **Warmup Ratio**: `0.05`
- **Weight Decay**: `0.01`
- **Seed**: `42`

## 3. 학습 테크닉 및 최적화 기법 (Training Optimization)
- **Completion Only Loss**:
  - `DataCollatorForCompletionOnlyLM` 사용 활성화 (`completion_only_loss: true`).
  - Qwen ChatML response template (`"<|im_start|>assistant\n"`) 기준 이전의 프롬프트 토큰은 손실(Loss) 계산에서 제외(`-100` 마스킹)하여 에이전트의 답변 토큰만 학습합니다.
- **Gradient Checkpointing**: 활성화 (`gradient_checkpointing: true`). 
- **Quantization**: 
  - 4-bit 양자화 로딩 (`load_in_4bit: true`).
  - NF4 Quantization (`bnb_4bit_quant_type: "nf4"`).
  - Double Quantization (`bnb_4bit_use_double_quant: true`).
  - Compute Data Type: `bfloat16`.
- **Flash Attention**: 
  - 하드웨어 가속(RTX 40 Ada 이상) 시 자동으로 작동되나 설정 파일 상 명시적인 `attn_implementation` 세팅은 누락되어 있음. (Transformers 기본 탑재 버전 사용)
- **Optimizer & Scheduler**:
  - **Optimizer**: 기본 `AdamW` (Transformers Default)
  - **Scheduler**: Linear Scheduler (Transformers Default, `warmup_ratio=0.05`)

## 4. PEFT / LoRA 설정
- **Type**: QLoRA (NF4 4-bit + LoRA 어댑터)
- **Rank (r)**: 32
- **Alpha (\(\alpha\))**: 64
- **Dropout**: 0.05
- **Target Modules**: 
  - `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Bias**: `none`
- **Task Type**: `CAUSAL_LM`
