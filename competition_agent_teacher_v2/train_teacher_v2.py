#!/usr/bin/env python3
"""
Teacher v2 학습 스크립트 — Phase 1 최적화 캠페인

적용 최적화:
  [Task 2] rsLoRA     : LoraConfig(use_rslora=True)
  [Task 3] LoRA+      : A:1e-4, B:4e-4 (TRL native or custom optimizer groups)
  [Task 4] Target Mod : q/k/v/o + gate/up/down (7개, 기존 동일)
  [Task 5] Cosine     : lr_scheduler_type="cosine"
  [Task 6] NEFTune    : neftune_noise_alpha=5
  [Task 7] 유지       : QLoRA / Completion-Only / BF16 / GradCkpt / FlashAttn

원칙:
  - 기존 production 코드 (train.py, configs/, adapters/) 절대 수정 금지
  - 기존 outputs/ 절대 덮어쓰기 금지
  - 기존 데이터/프롬프트/어댑터 수정 금지

실행:
  cd ~/LLM_Decision
  python competition_agent_teacher_v2/train_teacher_v2.py \\
      --config competition_agent_teacher_v2/config_teacher_v2.yaml
"""

import os
import sys
import json
import time
import logging
import argparse
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import torch

# 기존 코드 재사용 (수정 없이 import만)
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import Config
from configs.logging_config import setup_logging, get_logger
from train import ModelTrainer  # _prepare_qlora_model, prepare_datasets 재사용

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Task 12: 실시간 모니터링 Callback
# ══════════════════════════════════════════════════════════════════════════════

class TeacherV2MonitorCallback:
    """
    Transformers TrainerCallback 형태의 실시간 모니터링.
    출력 항목: Stage / Elapsed / ETA / GPU / RAM / LR / Loss
    """

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.t_start = time.time()
        self._last_loss: float = float("nan")
        self._last_eval_loss: float = float("nan")

    def _gpu_stats(self) -> Dict[str, str]:
        if not torch.cuda.is_available():
            return {}
        try:
            idx = torch.cuda.current_device()
            free, total = torch.cuda.mem_get_info(idx)
            used = total - free
            util = torch.cuda.utilization(idx)
            return {
                "gpu_util": f"{util}%",
                "gpu_mem":  f"{used/1024**3:.1f}/{total/1024**3:.1f}GB",
            }
        except Exception:
            return {}

    def _cpu_stats(self) -> Dict[str, str]:
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0.1)
            ram = psutil.virtual_memory()
            return {
                "cpu": f"{cpu:.0f}%",
                "ram": f"{ram.used/1024**3:.1f}/{ram.total/1024**3:.1f}GB",
            }
        except ImportError:
            return {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        """HuggingFace Trainer callback hook."""
        if logs is None:
            return

        step = state.global_step
        elapsed = time.time() - self.t_start
        eta = (elapsed / max(step, 1)) * (self.total_steps - step)

        loss = logs.get("loss", self._last_loss)
        eval_loss = logs.get("eval_loss", self._last_eval_loss)
        lr = logs.get("learning_rate", 0.0)

        if "loss" in logs:
            self._last_loss = loss
        if "eval_loss" in logs:
            self._last_eval_loss = eval_loss

        gpu = self._gpu_stats()
        cpu = self._cpu_stats()

        line = (
            f"[Step {step:>5}/{self.total_steps}] "
            f"Elapsed={timedelta(seconds=int(elapsed))} "
            f"ETA={timedelta(seconds=int(eta))} | "
            f"LR={lr:.2e} "
            f"Loss={loss:.4f} "
            f"EvalLoss={eval_loss:.4f} | "
            + " ".join(f"{k}={v}" for k, v in {**gpu, **cpu}.items())
        )
        logger.info(line)

    # Trainer expects these methods
    def on_train_begin(self, args, state, control, **kwargs):
        logger.info("=" * 70)
        logger.info("[teacher_v2] 학습 시작")
        logger.info(f"  Total Steps : {self.total_steps}")
        logger.info(f"  Start Time  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.t_start
        logger.info("=" * 70)
        logger.info(f"[teacher_v2] 학습 완료  소요: {timedelta(seconds=int(elapsed))}")
        logger.info("=" * 70)

    def on_evaluate(self, args, state, control, **kwargs):
        logger.info(f"[teacher_v2] Validation @ step {state.global_step}")

    def on_save(self, args, state, control, **kwargs):
        logger.info(f"[teacher_v2] Checkpoint 저장 @ step {state.global_step}")

    def on_step_end(self, args, state, control, **kwargs):
        pass

    def on_epoch_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.t_start
        epoch = state.epoch
        logger.info(
            f"[teacher_v2] Epoch {epoch:.0f} 완료 | "
            f"경과: {timedelta(seconds=int(elapsed))}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Teacher v2 Trainer
# ══════════════════════════════════════════════════════════════════════════════

class TeacherV2Trainer(ModelTrainer):
    """
    기존 ModelTrainer를 상속, 최소 오버라이드로 v2 최적화 적용.

    오버라이드 범위:
      - _prepare_qlora_model : rsLoRA (use_rslora=True) 추가
      - _setup_trainer       : Cosine / NEFTune / LoRA+ / Monitoring 추가
    그 외 모든 로직(데이터 로드, save, etc.)은 부모 그대로 사용.
    """

    def __init__(self, config: Config, resume_from_checkpoint: Optional[str] = None):
        super().__init__(config, resume_from_checkpoint)
        self._training_start_time: Optional[float] = None
        self._metrics_path = Path(config.paths.outputs_dir) / "training_metrics.json"
        self._metrics: Dict[str, Any] = {}

    # ── [Task 2] rsLoRA ──────────────────────────────────────────────────────
    def _prepare_qlora_model(self) -> None:
        """QLoRA + rsLoRA."""
        from transformers import BitsAndBytesConfig
        from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.optimization.load_in_4bit,
            bnb_4bit_use_double_quant=self.config.optimization.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=self.config.optimization.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(
                torch, self.config.optimization.bnb_4bit_compute_dtype
            ),
        )

        # Flash Attention 2 (설치 안 된 경우 자동 fallback)
        attn_impl = getattr(self.config.model, "attn_implementation", "eager")
        load_kwargs = dict(
            quantization_config=bnb_config,
            device_map=self.config.model.device_map,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        try:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model.name,
                attn_implementation=attn_impl,
                **load_kwargs,
            )
            self.logger.info(f"✓ Flash Attention: {attn_impl}")
        except (ValueError, ImportError) as e:
            self.logger.warning(f"Flash Attention 로드 실패 ({e}), eager fallback")
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model.name, **load_kwargs
            )

        self.model = prepare_model_for_kbit_training(self.model)

        if self.config.training.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # [Task 2] rsLoRA: use_rslora=True
        use_rslora = getattr(self.config.optimization, "use_rslora", False)
        peft_config = LoraConfig(
            r=self.config.optimization.lora_rank,
            lora_alpha=self.config.optimization.lora_alpha,
            lora_dropout=self.config.optimization.lora_dropout,
            target_modules=self.config.optimization.target_modules,
            bias=self.config.optimization.bias,
            task_type=TaskType.CAUSAL_LM,
            use_rslora=use_rslora,          # ← rsLoRA 핵심
        )
        self.model = get_peft_model(self.model, peft_config)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        self.logger.info(
            f"✓ QLoRA+rsLoRA | Trainable: {trainable:,}/{total:,} "
            f"({100*trainable/total:.3f}%)"
        )
        self.logger.info(
            f"  rsLoRA scaling = alpha/sqrt(r) = "
            f"{self.config.optimization.lora_alpha}/sqrt({self.config.optimization.lora_rank}) "
            f"= {self.config.optimization.lora_alpha / self.config.optimization.lora_rank**0.5:.4f}"
        )

    # ── [Tasks 3,5,6,12] Trainer 설정 오버라이드 ─────────────────────────────
    def _setup_trainer(self, train_dataset, val_dataset) -> None:
        """
        부모 _setup_trainer를 대체.
        추가: Cosine LR / NEFTune / LoRA+ / MonitorCallback
        기존과 동일: CompletionOnlyLoss / formatting_func / packing=False
        """
        from trl import SFTTrainer, SFTConfig

        tokenizer = self.tokenizer

        # ── Completion-Only Loss (기존 로직 그대로) ──────────────────────────
        DataCollatorForCompletionOnlyLM = self._load_completion_collator()
        data_collator = None
        if getattr(self.config.training, "completion_only_loss", True):
            response_template = "<|im_start|>assistant\n"
            response_template_ids = tokenizer.encode(
                response_template, add_special_tokens=False
            )
            self.logger.info(
                f"Completion-only loss: {repr(response_template)} "
                f"→ token_ids={response_template_ids}"
            )
            data_collator = DataCollatorForCompletionOnlyLM(
                response_template=response_template_ids,
                tokenizer=tokenizer,
                mlm=False,
            )

        # ── formatting_func (기존 그대로) ────────────────────────────────────
        def formatting_func(example):
            try:
                return tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
            except TypeError:
                return tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )

        # ── [Task 5] Cosine Scheduler ─────────────────────────────────────────
        lr_scheduler_type = getattr(
            self.config.training, "lr_scheduler_type", "cosine"
        )

        # ── [Task 6] NEFTune ──────────────────────────────────────────────────
        neftune_noise_alpha = getattr(
            self.config.training, "neftune_noise_alpha", None
        )

        # ── [Task 3] LoRA+ : TRL native 우선, fallback은 on_train_begin에서 처리 ──
        loraplus_lr_ratio = getattr(
            self.config.training, "loraplus_lr_ratio", None
        )

        # ── SFTConfig 구성 ────────────────────────────────────────────────────
        sft_kwargs = dict(
            output_dir=self.config.training.output_dir,
            num_train_epochs=self.config.training.num_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_ratio=self.config.training.warmup_ratio,
            max_steps=self.config.training.max_steps,
            bf16=self.config.training.mixed_precision == "bf16",
            fp16=self.config.training.mixed_precision == "fp16",
            save_strategy=self.config.training.save_strategy,
            save_steps=self.config.training.save_steps,
            eval_strategy="steps" if val_dataset else "no",
            eval_steps=self.config.training.eval_steps if val_dataset else None,
            save_total_limit=self.config.training.save_total_limit,
            logging_steps=50,
            logging_dir=self.config.logging.tensorboard_dir,
            seed=self.config.training.seed,
            dataloader_pin_memory=True,
            dataloader_drop_last=True,
            packing=False,
            max_length=1024,
            # [Task 5] Cosine
            lr_scheduler_type=lr_scheduler_type,
            # [Task 6] NEFTune
            neftune_noise_alpha=neftune_noise_alpha,
        )

        # [Task 3] LoRA+ — TRL native (v0.9+)
        self._loraplus_native = False
        if loraplus_lr_ratio is not None:
            try:
                import inspect
                if "loraplus_lr_ratio" in inspect.signature(SFTConfig.__init__).parameters:
                    sft_kwargs["loraplus_lr_ratio"] = loraplus_lr_ratio
                    self._loraplus_native = True
                    self.logger.info(
                        f"✓ LoRA+ (TRL native): ratio={loraplus_lr_ratio} "
                        f"→ A={self.config.training.learning_rate:.1e}, "
                        f"B={self.config.training.learning_rate*loraplus_lr_ratio:.1e}"
                    )
            except Exception:
                pass

        sft_config = SFTConfig(**sft_kwargs)

        # ── 총 step 수 계산 (모니터링용) ─────────────────────────────────────
        n_samples = len(train_dataset)
        steps_per_epoch = n_samples // (
            self.config.training.per_device_train_batch_size
            * self.config.training.gradient_accumulation_steps
        )
        total_steps = steps_per_epoch * self.config.training.num_epochs
        if self.config.training.max_steps > 0:
            total_steps = min(total_steps, self.config.training.max_steps)

        # ── [Task 12] Monitoring Callback ────────────────────────────────────
        monitor_cb = TeacherV2MonitorCallback(total_steps=total_steps)

        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=sft_config,
            formatting_func=formatting_func,
            data_collator=data_collator,
            callbacks=[monitor_cb],
        )

        # [Task 3] LoRA+ fallback — Custom Optimizer Groups
        if loraplus_lr_ratio is not None and not self._loraplus_native:
            self._apply_loraplus_optimizer(loraplus_lr_ratio)

        self.logger.info(f"✓ Trainer 설정 완료")
        self.logger.info(f"  LR Scheduler  : {lr_scheduler_type}")
        self.logger.info(f"  NEFTune alpha : {neftune_noise_alpha}")
        self.logger.info(f"  LoRA+ native  : {self._loraplus_native}")
        self.logger.info(f"  Total steps   : {total_steps}")

    def _apply_loraplus_optimizer(self, ratio: float) -> None:
        """
        [Task 3] LoRA+ fallback: Custom optimizer groups.
        TRL native 미지원 시 호출.
        A params: lr_a (config.training.learning_rate)
        B params: lr_b = lr_a * ratio
        others  : lr_a
        """
        from torch.optim import AdamW
        from transformers import get_cosine_schedule_with_warmup

        lr_a = getattr(
            self.config.training, "loraplus_lr_a", self.config.training.learning_rate
        )
        lr_b = getattr(
            self.config.training, "loraplus_lr_b", self.config.training.learning_rate * ratio
        )

        lora_a_params, lora_b_params, other_params = [], [], []
        for name, param in self.trainer.model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_A" in name:
                lora_a_params.append(param)
            elif "lora_B" in name:
                lora_b_params.append(param)
            else:
                other_params.append(param)

        optimizer = AdamW(
            [
                {"params": lora_a_params, "lr": lr_a, "name": "lora_A"},
                {"params": lora_b_params, "lr": lr_b, "name": "lora_B"},
                {"params": other_params,  "lr": lr_a, "name": "other"},
            ],
            weight_decay=self.config.training.weight_decay,
        )
        self.trainer.optimizer = optimizer

        # Cosine scheduler 재연결
        total_steps = self.trainer.args.max_steps
        if total_steps <= 0:
            steps_per_epoch = len(self.trainer.train_dataset) // (
                self.config.training.per_device_train_batch_size
                * self.config.training.gradient_accumulation_steps
            )
            total_steps = steps_per_epoch * self.config.training.num_epochs
        warmup_steps = int(total_steps * self.config.training.warmup_ratio)
        self.trainer.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )

        self.logger.info(
            f"✓ LoRA+ Custom Optimizer | A={lr_a:.1e} / B={lr_b:.1e} "
            f"(ratio={ratio})"
        )
        self.logger.info(
            f"  lora_A params: {len(lora_a_params)} | "
            f"lora_B params: {len(lora_b_params)} | "
            f"other params: {len(other_params)}"
        )

    def _load_completion_collator(self):
        """기존 train.py의 collator 로직을 그대로 복사 (production 코드 수정 안 함)."""
        DataCollatorForCompletionOnlyLM = None
        for _import in [
            "from trl import DataCollatorForCompletionOnlyLM",
            "from trl.trainer import DataCollatorForCompletionOnlyLM",
            "from trl.trainer.utils import DataCollatorForCompletionOnlyLM",
        ]:
            try:
                exec(_import, globals())
                DataCollatorForCompletionOnlyLM = globals()["DataCollatorForCompletionOnlyLM"]
                self.logger.info(f"✓ {_import}")
                break
            except (ImportError, KeyError):
                continue

        if DataCollatorForCompletionOnlyLM is None:
            self.logger.warning("DataCollatorForCompletionOnlyLM not found in trl, using fallback")

            class DataCollatorForCompletionOnlyLM:
                def __init__(self, response_template, tokenizer, mlm=False):
                    self.response_template_ids = (
                        response_template if isinstance(response_template, list)
                        else tokenizer.encode(response_template, add_special_tokens=False)
                    )
                    self.tokenizer = tokenizer
                    self.mlm = mlm

                def __call__(self, features):
                    def to_int_list(x):
                        return x.tolist() if hasattr(x, "tolist") else [int(v) for v in x]

                    max_len = max(len(to_int_list(f["input_ids"])) for f in features)
                    pad_id = self.tokenizer.pad_token_id or 0
                    batch_input_ids, batch_attention, batch_labels = [], [], []
                    for f in features:
                        ids    = to_int_list(f["input_ids"])
                        attn   = to_int_list(f.get("attention_mask", [1] * len(ids)))
                        labels = to_int_list(f["labels"]) if "labels" in f else list(ids)
                        tpl, tpl_len = self.response_template_ids, len(self.response_template_ids)
                        found = False
                        for i in range(len(ids) - tpl_len + 1):
                            if ids[i:i + tpl_len] == tpl:
                                for j in range(i + tpl_len):
                                    labels[j] = -100
                                found = True
                                break
                        if not found:
                            labels = [-100] * len(ids)
                        pad = max_len - len(ids)
                        ids   += [pad_id] * pad
                        attn  += [0]      * pad
                        labels += [-100]  * pad
                        batch_input_ids.append(ids)
                        batch_attention.append(attn)
                        batch_labels.append(labels)
                    return {
                        "input_ids":      torch.tensor(batch_input_ids,  dtype=torch.long),
                        "attention_mask": torch.tensor(batch_attention,  dtype=torch.long),
                        "labels":         torch.tensor(batch_labels,     dtype=torch.long),
                    }

        return DataCollatorForCompletionOnlyLM

    # ── 학습 실행 + 시간 측정 ────────────────────────────────────────────────
    def train(self) -> None:
        self.logger.info("=" * 80)
        self.logger.info("TEACHER v2 TRAINING PIPELINE")
        self.logger.info("=" * 80)

        self.logger.info("\n[1/3] Preparing model...")
        t0 = time.time()
        self.prepare_model()
        model_load_time = time.time() - t0
        self.logger.info(f"  모델 로드: {model_load_time:.1f}초")

        self.logger.info("\n[2/3] Preparing datasets...")
        train_dataset, val_dataset = self.prepare_datasets()

        self.logger.info("\n[3/3] Setting up trainer...")
        self._setup_trainer(train_dataset, val_dataset)

        self.logger.info("\n학습 시작...")
        t_train = time.time()
        self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
        training_time = time.time() - t_train

        self.logger.info(f"\n학습 완료: {training_time/3600:.2f}h")

        # 체크포인트 저장
        self.trainer.save_model(self.config.training.output_dir)
        self.tokenizer.save_pretrained(self.config.training.output_dir)

        # 학습 메트릭 저장 (compare_teacher_v2.py에서 사용)
        self._metrics.update({
            "model_name": self.config.model.name,
            "config_path": "competition_agent_teacher_v2/config_teacher_v2.yaml",
            "training_time_sec": training_time,
            "training_time_str": str(timedelta(seconds=int(training_time))),
            "model_load_time_sec": model_load_time,
            "global_step": self.trainer.state.global_step,
            "best_eval_loss": self.trainer.state.best_metric,
            "log_history": self.trainer.state.log_history[-10:],
            "optimizations": {
                "rsLoRA": True,
                "loraplus": True,
                "cosine_scheduler": True,
                "neftune": True,
                "target_modules": self.config.optimization.target_modules,
            },
            "timestamp": datetime.now().isoformat(),
        })
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._metrics_path, "w") as f:
            json.dump(self._metrics, f, indent=2, default=str)
        self.logger.info(f"✓ 학습 메트릭 저장: {self._metrics_path}")

        self._save_experiment()

        self.logger.info("\n" + "=" * 80)
        self.logger.info("✓ TEACHER v2 TRAINING COMPLETED")
        self.logger.info("=" * 80)
        self.logger.info("\n다음 단계:")
        self.logger.info(
            "  python competition_agent_teacher_v2/validate_teacher_v2.py "
            f"--checkpoint {self.config.training.output_dir}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Teacher v2 학습 (Phase 1 최적화)")
    parser.add_argument(
        "--config",
        default="competition_agent_teacher_v2/config_teacher_v2.yaml",
    )
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    # 기존 production 출력 경로 보호
    dangerous_paths = ["./outputs/checkpoints", "./outputs/quantized"]
    config_raw = Path(args.config).read_text()
    for path in dangerous_paths:
        if path in config_raw:
            logger.error(
                f"위험: config에 production 경로({path}) 감지! "
                "teacher_v2 전용 경로(outputs/teacher_v2/)를 사용하세요."
            )
            sys.exit(1)

    setup_logging(log_dir="./logs/teacher_v2", level="INFO")

    config = Config.from_yaml(args.config)
    config.create_directories()

    Path("./outputs/teacher_v2").mkdir(parents=True, exist_ok=True)
    Path("./logs/teacher_v2").mkdir(parents=True, exist_ok=True)

    trainer = TeacherV2Trainer(config, resume_from_checkpoint=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
