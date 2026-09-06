#!/usr/bin/env python3
"""
Training script with LoRA/QLoRA support.

Features:
- Transformers + TRL integration
- PEFT for LoRA/QLoRA
- Accelerate for distributed training
- BitsAndBytes for quantization
- Gradient checkpointing and mixed precision
- Auto resume from checkpoints
- TensorBoard monitoring
"""

import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

from configs.config import Config
from configs.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Universal model trainer with LoRA/QLoRA support."""
    
    def __init__(self, config: Config, resume_from_checkpoint: Optional[str] = None):
        """Initialize trainer."""
        self.config = config
        self.resume_from_checkpoint = resume_from_checkpoint
        self.logger = get_logger(self.__class__.__name__)
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def prepare_model(self) -> None:
        """Load and prepare model with appropriate configuration."""
        self.logger.info(f"Loading model: {self.config.model.name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.name,
            trust_remote_code=self.config.model.trust_remote_code,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.logger.info(f"✓ Tokenizer loaded (vocab size: {len(self.tokenizer)})")
        
        # Prepare model based on optimization type
        if self.config.optimization.type == "qlora":
            self._prepare_qlora_model()
        else:
            self._prepare_lora_model()
        
        self.logger.info(f"✓ Model prepared with {self.config.optimization.type.upper()}")
    
    def _prepare_lora_model(self) -> None:
        """Prepare model for LoRA fine-tuning."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.name,
            torch_dtype=getattr(torch, self.config.model.torch_dtype),
            device_map=self.config.model.device_map,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        
        # Enable gradient checkpointing
        if self.config.training.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Apply LoRA
        peft_config = LoraConfig(
            r=self.config.optimization.lora_rank,
            lora_alpha=self.config.optimization.lora_alpha,
            lora_dropout=self.config.optimization.lora_dropout,
            target_modules=self.config.optimization.target_modules,
            bias=self.config.optimization.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, peft_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(
            f"Trainable params: {trainable_params:,} / {all_params:,} "
            f"({100 * trainable_params / all_params:.2f}%)"
        )
    
    def _prepare_qlora_model(self) -> None:
        """Prepare model for QLoRA fine-tuning (4-bit quantization)."""
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.optimization.load_in_4bit,
            bnb_4bit_use_double_quant=self.config.optimization.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=self.config.optimization.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, self.config.optimization.bnb_4bit_compute_dtype),
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.name,
            quantization_config=bnb_config,
            device_map=self.config.model.device_map,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Enable gradient checkpointing
        if self.config.training.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Apply LoRA on top of quantized model
        peft_config = LoraConfig(
            r=self.config.optimization.lora_rank,
            lora_alpha=self.config.optimization.lora_alpha,
            lora_dropout=self.config.optimization.lora_dropout,
            target_modules=self.config.optimization.target_modules,
            bias=self.config.optimization.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, peft_config)
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(
            f"Trainable params: {trainable_params:,} / {all_params:,} "
            f"({100 * trainable_params / all_params:.2f}%)"
        )
    
    def prepare_datasets(self):
        """Load and prepare training datasets."""
        self.logger.info("Loading datasets...")
        
        train_path = Path(self.config.paths.data_processed) / "train.json"
        val_path = Path(self.config.paths.data_processed) / "validation.json"
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found at {train_path}")
        
        # Load datasets directly from list of dicts to prevent pyarrow type inference errors
        from datasets import Dataset
        
        self.logger.info("Loading training dataset from JSON list...")
        with open(train_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        
        # Pop 'tools' key as it's not used by SFTTrainer but causes pyarrow type errors
        for item in train_data:
            item.pop("tools", None)
            
        train_dataset = Dataset.from_list(train_data)
        
        val_dataset = None
        if val_path.exists():
            self.logger.info("Loading validation dataset from JSON list...")
            try:
                with open(val_path, "r", encoding="utf-8") as f:
                    val_data = json.load(f)
                for item in val_data:
                    item.pop("tools", None)
                val_dataset = Dataset.from_list(val_data)
            except Exception as e:
                self.logger.warning(f"Could not load validation data: {str(e)}")
        
        self.logger.info(f"✓ Loaded {len(train_dataset)} training examples")
        if val_dataset:
            self.logger.info(f"✓ Loaded {len(val_dataset)} validation examples")
        
        return train_dataset, val_dataset
    
    def train(self) -> None:
        """Execute training loop."""
        self.logger.info("=" * 80)
        self.logger.info("TRAINING PIPELINE")
        self.logger.info("=" * 80)
        
        # Prepare model
        self.logger.info("\n[1/3] Preparing model...")
        self.prepare_model()
        
        # Load datasets
        self.logger.info("\n[2/3] Preparing datasets...")
        train_dataset, val_dataset = self.prepare_datasets()
        
        # Setup training
        self.logger.info("\n[3/3] Setting up training...")
        self._setup_trainer(train_dataset, val_dataset)
        
        # Train
        self.logger.info("\nStarting training...")
        self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
        
        # Save final model and tokenizer
        self.logger.info(f"Saving final model and tokenizer to {self.config.training.output_dir}")
        self.trainer.save_model(self.config.training.output_dir)
        self.tokenizer.save_pretrained(self.config.training.output_dir)
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("✓ TRAINING COMPLETED")
        self.logger.info("=" * 80)
        
        # Save final model
        self._save_experiment()
    
    def _setup_trainer(self, train_dataset, val_dataset) -> None:
        """Setup SFT trainer with chat-template formatting.

        v2 변경사항:
        - DataCollatorForCompletionOnlyLM 추가:
          프롬프트(system+user) 토큰은 loss 계산에서 제외하고
          assistant 응답 토큰만 학습. 라벨 신호가 정확해져 수렴 품질 향상.
        - Qwen ChatML response_template: "<|im_start|>assistant\\n"
        """
        # DataCollatorForCompletionOnlyLM: TRL 버전마다 import 경로가 다름
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
            # TRL이 너무 오래됐거나 최신 API 변경 → 직접 구현
            self.logger.warning(
                "DataCollatorForCompletionOnlyLM을 trl에서 찾을 수 없음. "
                "직접 구현 사용 (동일한 동작)."
            )
            import torch
            from transformers import DataCollatorForSeq2Seq

            class DataCollatorForCompletionOnlyLM:
                """
                response_template_ids 이전 토큰을 -100으로 마스킹.
                SFTTrainer의 formatting_func과 함께 사용.
                """
                def __init__(self, response_template, tokenizer, mlm=False):
                    self.response_template_ids = (
                        response_template if isinstance(response_template, list)
                        else tokenizer.encode(response_template, add_special_tokens=False)
                    )
                    self.tokenizer = tokenizer
                    self.mlm = mlm

                def __call__(self, features):
                    import torch

                    def to_int_list(x):
                        """tensor/list 모두 int 리스트로 변환."""
                        if hasattr(x, "tolist"):
                            return x.tolist()
                        return [int(v) for v in x]

                    max_len = max(len(to_int_list(f["input_ids"])) for f in features)
                    pad_id = self.tokenizer.pad_token_id or 0

                    batch_input_ids, batch_attention, batch_labels = [], [], []
                    for f in features:
                        ids = to_int_list(f["input_ids"])
                        # attention_mask 없으면 모두 1 (TRL 버전 따라 없을 수 있음)
                        attn = to_int_list(f["attention_mask"]) if "attention_mask" in f else [1] * len(ids)
                        # labels 이미 있으면 사용, 없으면 input_ids 복사
                        labels = to_int_list(f["labels"]) if "labels" in f else list(ids)

                        # response_template 위치 찾기
                        tpl = self.response_template_ids
                        tpl_len = len(tpl)
                        found = False
                        for i in range(len(ids) - tpl_len + 1):
                            if ids[i:i + tpl_len] == tpl:
                                # template 포함 이전 모든 토큰 마스킹
                                for j in range(i + tpl_len):
                                    labels[j] = -100
                                found = True
                                break
                        if not found:
                            # template 없으면 전체 마스킹 (학습 기여 안 함)
                            labels = [-100] * len(ids)

                        # 패딩
                        pad_len = max_len - len(ids)
                        ids   += [pad_id] * pad_len
                        attn  += [0]     * pad_len
                        labels += [-100] * pad_len

                        batch_input_ids.append(ids)
                        batch_attention.append(attn)
                        batch_labels.append(labels)

                    return {
                        "input_ids":      torch.tensor(batch_input_ids,  dtype=torch.long),
                        "attention_mask": torch.tensor(batch_attention,  dtype=torch.long),
                        "labels":         torch.tensor(batch_labels,     dtype=torch.long),
                    }

        tokenizer = self.tokenizer  # closure 캡처용

        def formatting_func(example):
            """Apply tokenizer chat template to convert messages list → training string.
            TRL SFTTrainer calls this with a single example (batched=False).
            enable_thinking=False: Gemma 4 전용 파라미터 (다른 모델은 무시됨)
            """
            try:
                return tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,   # Gemma 4: 추론 토큰 비활성화
                )
            except TypeError:
                # enable_thinking 미지원 모델(Qwen 등) fallback
                return tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )

        # ── Completion-only loss 설정 ──────────────────────────────────────────
        # Qwen ChatML에서 assistant 응답은 "<|im_start|>assistant\n" 이후에 시작.
        # 이 토큰들을 response_template으로 지정하면 그 이전 토큰들(-100 masking)은
        # loss 계산에서 제외됨 → 프롬프트가 아닌 라벨 학습만 수행.
        #
        # 토큰 ID 방식 사용: 문자열 방식은 토크나이저에 따라 불일치 위험 있음.
        # "<|im_start|>assistant\n"을 토크나이즈하면 Qwen에서 [151644, 77091, 198]
        # (실제 값은 토크나이저마다 다를 수 있으므로 런타임에 계산)
        use_completion_only = getattr(self.config.training, "completion_only_loss", True)

        data_collator = None
        if use_completion_only:
            response_template = "<|im_start|>assistant\n"
            response_template_ids = tokenizer.encode(
                response_template, add_special_tokens=False
            )
            self.logger.info(
                f"Completion-only loss 활성화: response_template={repr(response_template)} "
                f"→ token_ids={response_template_ids}"
            )
            data_collator = DataCollatorForCompletionOnlyLM(
                response_template=response_template_ids,
                tokenizer=tokenizer,
                mlm=False,
            )
        else:
            self.logger.info("Completion-only loss 비활성화: 전체 시퀀스 loss 사용")

        sft_config = SFTConfig(
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
            # SFTConfig specific parameters
            packing=False,
            max_length=1024,
            # dataset_text_field 제거 → formatting_func 사용
        )

        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=sft_config,
            formatting_func=formatting_func,
            data_collator=data_collator,   # None이면 기본 collator 사용
        )
    
    def _save_experiment(self) -> None:
        """Save experiment metadata."""
        experiment_dir = Path(self.config.training.output_dir) / "experiment"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = experiment_dir / "config.yaml"
        self.config.save(str(config_path))
        
        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "model": self.config.model.name,
            "dataset": self.config.dataset.name,
            "optimization": self.config.optimization.type,
            "training_steps": self.trainer.state.global_step,
        }
        
        metadata_path = experiment_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"✓ Experiment saved to {experiment_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train model with LoRA/QLoRA")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lora.yaml",
        help="Config file path (base.yaml, lora.yaml, or qlora.yaml)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_dir="./logs", level="INFO")
    
    # Load config
    config = Config.from_yaml(args.config)
    config.create_directories()
    
    # Train
    trainer = ModelTrainer(config, resume_from_checkpoint=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
