#!/usr/bin/env python3
"""
Production Optimization Campaign (Phase 1)
Qwen2.5-1.5B-Instruct 최적화 학습 스크립트.

특징:
- 기존 train.py 및 SFTTrainer 호환성 유지 및 최소 변경 구현
- LoRA+ 구현 (Optimizer Parameter Grouping: A=1e-4, B=4e-4)
- rsLoRA 및 NEFTune, Early Stopping (patience=2) 통합
- 엄격한 난수 시드 고정 (Python, Numpy, PyTorch)
- 실시간 리소스 모니터링 Callback (CPU, RAM, GPU, Elapsed/Remaining Time)
- 검증 매 스텝 종료 시 결과 데이터 (Predictions, Probabilities, Confusion Matrix) 저장
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# torchvision Mocking (transforms, functional, io) to bypass 구버전 torchvision 0.2.0 import bug
from types import ModuleType

# 1. torchvision.transforms.functional.pil_to_tensor Mocking
try:
    import torchvision.transforms.functional as tv_func
    if not hasattr(tv_func, "pil_to_tensor"):
        tv_func.pil_to_tensor = lambda *args, **kwargs: None
except ImportError:
    func_mock = ModuleType("torchvision.transforms.functional")
    func_mock.pil_to_tensor = lambda *args, **kwargs: None
    sys.modules["torchvision.transforms.functional"] = func_mock

# 2. torchvision.transforms.InterpolationMode Mocking
try:
    import torchvision.transforms as tv_trans
    if not hasattr(tv_trans, "InterpolationMode"):
        class MetaInterpolationMode(type):
            def __getattr__(cls, name):
                return name.lower()
        class DummyInterpolationMode(metaclass=MetaInterpolationMode):
            NEAREST = "nearest"
            BILINEAR = "bilinear"
            BICUBIC = "bicubic"
            NEAREST_EXACT = "nearest_exact"
        tv_trans.InterpolationMode = DummyInterpolationMode
except ImportError:
    trans_mock = ModuleType("torchvision.transforms")
    class MetaInterpolationMode(type):
        def __getattr__(cls, name):
            return name.lower()
    class DummyInterpolationMode(metaclass=MetaInterpolationMode):
        NEAREST = "nearest"
        BILINEAR = "bilinear"
        BICUBIC = "bicubic"
        NEAREST_EXACT = "nearest_exact"
    trans_mock.InterpolationMode = DummyInterpolationMode
    sys.modules["torchvision.transforms"] = trans_mock

# 3. torchvision.io (ImageReadMode, decode_image) Mocking
try:
    from torchvision.io import ImageReadMode, decode_image
except ImportError:
    import torchvision
    io_mock = ModuleType("torchvision.io")
    io_mock.ImageReadMode = object
    io_mock.decode_image = lambda *args, **kwargs: None
    sys.modules["torchvision.io"] = io_mock
    if not hasattr(torchvision, "io"):
        torchvision.io = io_mock

# 4. torchvision.transforms.v2 & v2.functional Mocking
try:
    import torchvision.transforms.v2
except ImportError:
    v2_mock = ModuleType("torchvision.transforms.v2")
    sys.modules["torchvision.transforms.v2"] = v2_mock
    
    v2_func_mock = ModuleType("torchvision.transforms.v2.functional")
    sys.modules["torchvision.transforms.v2.functional"] = v2_func_mock
    
    v2_mock.functional = v2_func_mock

import json
import time
import yaml
import random
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    EarlyStoppingCallback,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# 기존 프로젝트 config 로더 가져오기
from configs.config import Config
from configs.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


# ── 난수 시드 고정 ───────────────────────────────────────────────────────────
def fix_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"✓ All random seeds fixed to {seed}")


# ── 리소스 모니터링 헬퍼 ──────────────────────────────────────────────────────
def get_resource_metrics() -> Dict[str, Any]:
    metrics = {
        "gpu_util": 0,
        "gpu_mem_used": 0.0,
        "gpu_mem_total": 0.0,
        "cpu_util": 0.0,
        "ram_util": 0.0,
    }
    
    # GPU 모니터링 (nvidia-smi 파싱)
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,nounits,noheader",
            ],
            capture_output=True,
            text=True,
            check=True
        )
        lines = res.stdout.strip().split("\n")
        if lines:
            parts = [p.strip() for p in lines[0].split(",")]
            metrics["gpu_util"] = int(parts[0])
            metrics["gpu_mem_used"] = float(parts[1])
            metrics["gpu_mem_total"] = float(parts[2])
    except Exception:
        pass

    # CPU / RAM 모니터링
    try:
        import psutil
        metrics["cpu_util"] = psutil.cpu_percent()
        metrics["ram_util"] = psutil.virtual_memory().percent
    except ImportError:
        # psutil 미설치 시 fallback
        try:
            res_cpu = subprocess.run(
                ["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True
            )
            metrics["cpu_util"] = float(res_cpu.stdout.strip().split()[0]) * 10.0
        except Exception:
            pass
        metrics["ram_util"] = 0.0
        
    return metrics


# ── LoRA+ 적용 SFTTrainer ─────────────────────────────────────────────────────
class LoRAPlusSFTTrainer(SFTTrainer):
    """
    SFTTrainer 상속 후 create_optimizer를 오버라이드하여
    LoRA A (1e-4)와 LoRA B (4e-4)의 학습률을 차등 적용합니다.
    """
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        logger.info("Setting up LoRA+ Custom Optimizer groups...")
        
        lora_a_params = []
        lora_b_params = []
        other_params = []

        # 파라미터 분류
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_A" in name or "matrix_A" in name:
                lora_a_params.append(param)
            elif "lora_B" in name or "matrix_B" in name:
                lora_b_params.append(param)
            else:
                other_params.append(param)

        optimizer_grouped_parameters = []
        if lora_a_params:
            optimizer_grouped_parameters.append({
                "params": lora_a_params,
                "lr": 1.0e-4,  # LoRA A LR
                "weight_decay": self.args.weight_decay,
            })
            logger.info(f"  ✓ LoRA A Parameters Group: {len(lora_a_params)} tensors (LR: 1.0e-4)")
            
        if lora_b_params:
            optimizer_grouped_parameters.append({
                "params": lora_b_params,
                "lr": 4.0e-4,  # LoRA B LR
                "weight_decay": self.args.weight_decay,
            })
            logger.info(f"  ✓ LoRA B Parameters Group: {len(lora_b_params)} tensors (LR: 4.0e-4)")
            
        if other_params:
            optimizer_grouped_parameters.append({
                "params": other_params,
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
            })
            logger.info(f"  ✓ Other Parameters Group: {len(other_params)} tensors (LR: {self.args.learning_rate})")

        from torch.optim import AdamW
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            eps=self.args.adam_epsilon,
            betas=(self.args.adam_beta1, self.args.adam_beta2)
        )
        return self.optimizer


# ── 실시간 리소스 로깅 및 Validation 기록 Callback ──────────────────────────────
class RealTimeMonitoringCallback(TrainerCallback):
    """실시간 리소스(CPU, RAM, GPU) 사용량 및 학습 경과 정보를 지속 모니터링하여 출력."""
    def __init__(self, start_time: float, output_dir: Path, tokenizer, validation_dataset):
        self.start_time = start_time
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.val_dataset = validation_dataset
        self.last_eval_step = 0

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # 10 step마다 모니터링 로그 출력
        if state.global_step % 10 == 0 or state.global_step == 1:
            elapsed = time.time() - self.start_time
            # 잔여 시간 계산
            est_remaining = 0.0
            if state.global_step > 0 and state.max_steps > 0:
                est_remaining = (elapsed / state.global_step) * (state.max_steps - state.global_step)
            elif state.global_step > 0 and state.num_train_epochs > 0:
                # epoch 기반 스텝 유추
                total_steps = state.max_steps if state.max_steps > 0 else (state.global_step / (state.epoch or 1.0)) * state.num_train_epochs
                est_remaining = (elapsed / state.global_step) * (total_steps - state.global_step)

            res = get_resource_metrics()
            
            lr = 0.0
            if state.log_history:
                for log in reversed(state.log_history):
                    if "learning_rate" in log:
                        lr = log["learning_rate"]
                        break
            
            loss = 0.0
            if state.log_history:
                for log in reversed(state.log_history):
                    if "loss" in log:
                        loss = log["loss"]
                        break

            val_loss = "N/A"
            if state.log_history:
                for log in reversed(state.log_history):
                    if "eval_loss" in log:
                        val_loss = f"{log['eval_loss']:.4f}"
                        break

            print(
                f"[Step {state.global_step}] "
                f"Elapsed: {elapsed/60:.1f}m | Est.Remaining: {est_remaining/60:.1f}m | "
                f"GPU Util: {res['gpu_util']}% | GPU Mem: {res['gpu_mem_used']:.0f}/{res['gpu_mem_total']:.0f}MB | "
                f"CPU: {res['cpu_util']:.1f}% | RAM: {res['ram_util']:.1f}% | "
                f"LR: {lr:.2e} | Loss: {loss:.4f} | ValLoss: {val_loss}",
                flush=True
            )

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        # validation이 수행될 때마다 예측 데이터 생성 및 저장 (Predictions, Probabilities, Confusion Matrix)
        if state.global_step == self.last_eval_step:
            return
        self.last_eval_step = state.global_step
        
        logger.info(f"Saving validation outputs at step {state.global_step}...")
        model = kwargs.get("model")
        if model is None or self.val_dataset is None:
            return

        model.eval()
        predictions = []
        probabilities = []
        confidences = []
        ground_truth = []
        ids = []

        # 14개 클래스 액션
        ACTIONS = [
            "read_file", "write_file", "edit_file", "apply_patch",
            "grep_search", "glob_pattern", "list_directory",
            "run_bash", "run_tests", "lint_or_typecheck",
            "web_search", "ask_user", "plan_task", "respond_only",
        ]
        
        # evaluation을 위해 미니배치로 추론 수행
        with torch.no_grad():
            for i, example in enumerate(self.val_dataset):
                try:
                    # formatting_func와 동일하게 chat_template 적용
                    prompt = self.tokenizer.apply_chat_template(
                        example["messages"],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
                    outputs = model(**inputs)
                    
                    # 마지막 토큰의 로짓 획득
                    logits = outputs.logits[0, -1, :]
                    
                    # ACTIONS 토큰들에 대한 확률만 소프트맥스 적용
                    action_logits = []
                    for action in ACTIONS:
                        # 각 액션 단어의 토큰 ID에 대한 로짓 추출 (안전하게 encode 사용)
                        token_ids = self.tokenizer.encode(action, add_special_tokens=False)
                        if token_ids:
                            action_logits.append(logits[token_ids[0]].item())
                        else:
                            action_logits.append(-9999.0)
                            
                    # softmax
                    exp_logits = np.exp(action_logits - np.max(action_logits))
                    probs = exp_logits / exp_logits.sum()
                    
                    max_idx = np.argmax(probs)
                    pred_label = ACTIONS[max_idx]
                    prob_val = probs[max_idx]
                    
                    # Ground Truth 찾기
                    gt_label = "respond_only"
                    for msg in reversed(example["messages"]):
                        if msg["role"] == "assistant":
                            # 실제 예측 대상 레이블 파싱
                            content = msg["content"].strip()
                            if content in ACTIONS:
                                gt_label = content
                            break

                    predictions.append(pred_label)
                    probabilities.append(probs.tolist())
                    confidences.append(prob_val)
                    ground_truth.append(gt_label)
                    ids.append(example.get("id", f"val_{i}"))
                    
                except Exception as e:
                    logger.warning(f"Error evaluating sample {i}: {str(e)}")

        # Confusion Matrix 계산
        from collections import Counter
        cm = {act: {a: 0 for a in ACTIONS} for act in ACTIONS}
        for gt, pred in zip(ground_truth, predictions):
            if gt in cm and pred in cm[gt]:
                cm[gt][pred] += 1

        # 결과 파일 저장
        val_output_dir = self.output_dir / "validation"
        val_output_dir.mkdir(parents=True, exist_ok=True)
        
        step_data = {
            "step": state.global_step,
            "epoch": state.epoch,
            "metrics": metrics,
            "results": [
                {
                    "id": idx,
                    "ground_truth": gt,
                    "prediction": pred,
                    "confidence": conf,
                    "probabilities": prob
                }
                for idx, gt, pred, conf, prob in zip(ids, ground_truth, predictions, confidences, probabilities)
            ],
            "confusion_matrix": cm
        }
        
        out_path = val_output_dir / f"eval_results_step_{state.global_step}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(step_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Validation outputs saved to {out_path}")
        
        # 모델을 다시 학습 모드로 전환
        model.train()


# ── 트레이너 메인 오케스트레이터 ────────────────────────────────────────────────
class ModelTrainerV2:
    def __init__(self, config_path: str, resume_from_checkpoint: bool = False, max_steps: int = -1, rank: int = 32):
        self.config_path = config_path
        self.resume_from_checkpoint = resume_from_checkpoint
        self.max_steps = max_steps
        self.rank = rank
        self.start_time = time.time()
        
        # 1. YAML 파싱 및 특수 파라미터 분리
        with open(config_path, "r", encoding="utf-8") as f:
            self.raw_yaml = yaml.safe_load(f)
            
        # config.py와 충돌을 예방하기 위해 extends 병합을 자체 수행 후 
        # 특수 필드들을 백업하고 pop하여 config 파서로 넘겨줌
        self.extra_training = self.raw_yaml.pop("extra_training_args", {})
        self.extra_optimization = self.raw_yaml.pop("extra_optimization_args", {})
        
        # 임시 yaml 저장하여 기존 Config 모듈로 파싱
        # (base_config 상대 경로 해석 오류 방지를 위해 원본과 같은 디렉토리에 생성)
        temp_dir = Path(config_path).parent
        self.temp_config_path = str(temp_dir / f"temp_{Path(config_path).name}")
        
        with open(self.temp_config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.raw_yaml, f)
            
        try:
            self.config = Config.from_yaml(self.temp_config_path)
        finally:
            # 파싱 후 임시 파일 정리
            try:
                os.remove(self.temp_config_path)
            except OSError:
                pass
                
        self.config.create_directories()
        
        # Rank 덮어쓰기 적용 (Ablation 및 YAML Override 대응)
        self.config.optimization.lora_rank = self.rank
        self.config.optimization.lora_alpha = self.rank * 2
        
        # 2. 시드 고정
        seed = self.config.training.seed
        fix_all_seeds(seed)
        
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def prepare_model(self) -> None:
        logger.info(f"Loading base model for Campaign V2: {self.config.model.name}")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.name,
            trust_remote_code=self.config.model.trust_remote_code,
            padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # QLoRA 4-bit 양자화 설정 적용
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
        
        self.model = prepare_model_for_kbit_training(self.model)
        
        if self.config.training.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # PEFT / LoRA Config (rsLoRA 지원 포함)
        use_rslora = self.extra_optimization.get("use_rslora", True)
        logger.info(f"Applying LoRA (rank={self.rank}, alpha={self.rank*2}, use_rslora={use_rslora})")
        
        peft_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.rank * 2,
            lora_dropout=self.config.optimization.lora_dropout,
            target_modules=self.config.optimization.target_modules,
            bias=self.config.optimization.bias,
            task_type=TaskType.CAUSAL_LM,
            use_rslora=use_rslora,               # rsLoRA 명시적 적용
        )
        
        self.model = get_peft_model(self.model, peft_config)
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"✓ Trainable params: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")

    def prepare_datasets(self) -> Tuple[Dataset, Dataset]:
        logger.info("Loading datasets...")
        train_path = Path(self.config.paths.data_processed) / "train.json"
        val_path = Path(self.config.paths.data_processed) / "validation.json"

        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found at {train_path}")

        logger.info("Loading training dataset from JSON list...")
        with open(train_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        for item in train_data:
            item.pop("tools", None)
        train_dataset = Dataset.from_list(train_data)

        val_dataset = None
        if val_path.exists():
            logger.info("Loading validation dataset from JSON list...")
            with open(val_path, "r", encoding="utf-8") as f:
                val_data = json.load(f)
            for item in val_data:
                item.pop("tools", None)
            val_dataset = Dataset.from_list(val_data)

        return train_dataset, val_dataset

    def train(self) -> None:
        self.prepare_model()
        train_dataset, val_dataset = self.prepare_datasets()
        
        # ── SFTConfig 설정 ──────────────────────────────────────────────────────
        # TRL의 DataCollatorForCompletionOnlyLM 바인딩
        response_template = "<|im_start|>assistant\n"
        response_template_ids = self.tokenizer.encode(
            response_template, add_special_tokens=False
        )
        
        DataCollatorForCompletionOnlyLM = None
        for _import in [
            "from trl import DataCollatorForCompletionOnlyLM",
            "from trl.trainer import DataCollatorForCompletionOnlyLM",
            "from trl.trainer.utils import DataCollatorForCompletionOnlyLM",
        ]:
            try:
                exec(_import, globals())
                DataCollatorForCompletionOnlyLM = globals()["DataCollatorForCompletionOnlyLM"]
                logger.info(f"✓ {_import}")
                break
            except (ImportError, KeyError):
                continue

        if DataCollatorForCompletionOnlyLM is None:
            logger.warning("DataCollatorForCompletionOnlyLM not found in trl. Using custom fallback implementation.")
            import torch
            class DataCollatorForCompletionOnlyLM:
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
                        if hasattr(x, "tolist"): return x.tolist()
                        return [int(v) for v in x]
                    
                    max_len = max(len(to_int_list(f["input_ids"])) for f in features)
                    pad_id = self.tokenizer.pad_token_id or 0

                    batch_input_ids, batch_attention, batch_labels = [], [], []
                    for f in features:
                        ids = to_int_list(f["input_ids"])
                        attn = to_int_list(f["attention_mask"]) if "attention_mask" in f else [1] * len(ids)
                        labels = to_int_list(f["labels"]) if "labels" in f else list(ids)

                        tpl = self.response_template_ids
                        tpl_len = len(tpl)
                        found = False
                        for i in range(len(ids) - tpl_len + 1):
                            if ids[i:i + tpl_len] == tpl:
                                for j in range(i + tpl_len):
                                    labels[j] = -100
                                found = True
                                break
                        if not found:
                            labels = [-100] * len(ids)

                        pad_len = max_len - len(ids)
                        ids += [pad_id] * pad_len
                        attn += [0] * pad_len
                        labels += [-100] * pad_len

                        batch_input_ids.append(ids)
                        batch_attention.append(attn)
                        batch_labels.append(labels)

                    return {
                        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                        "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
                        "labels": torch.tensor(batch_labels, dtype=torch.long),
                    }

        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template_ids,
            tokenizer=self.tokenizer,
            mlm=False,
        )

        def formatting_func(example):
            return self.tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )

        # Early Stopping 및 Best model params 로드
        patience = self.extra_training.get("patience", 2)
        load_best = self.extra_training.get("load_best_model_at_end", True)
        metric_best = self.extra_training.get("metric_for_best_model", "eval_macro_f1")
        greater_better = self.extra_training.get("greater_is_better", True)
        
        # Smoke Test용 max_steps 오버라이드
        max_steps = self.max_steps if self.max_steps > 0 else self.config.training.max_steps

        # 메모리 폭발 방지 (OOM): Validation 시 전체 Logits을 GPU/RAM에 누적하지 않고 argmax 결과만 저장
        def preprocess_logits_for_metrics(logits, labels):
            import torch
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)

        def compute_metrics(eval_pred):
            # eval_pred.predictions는 이미 preprocess_logits_for_metrics를 거쳐
            # argmax 인덱스 배열(preds) 형태로 들어옴.
            preds, labels = eval_pred
            if isinstance(preds, tuple):
                preds = preds[0]

            # Causal LM: preds[i]는 position i+1의 토큰을 예측.
            # labels와 정렬하려면 preds를 1칸 왼쪽으로 shift해야 함.
            # preds: [p0, p1, ..., pN-1] → shifted_preds: [p0, p1, ..., pN-2]
            # labels: [l0, l1, ..., lN-1] → shifted_labels: [l1, l2, ..., lN-1]
            shifted_preds = preds[:, :-1]
            shifted_labels = labels[:, 1:]

            # loss 계산 대상 토큰(-100이 아닌 것)만 추출
            mask = shifted_labels != -100
            flat_preds = shifted_preds[mask].flatten()
            flat_labels = shifted_labels[mask].flatten()

            if len(flat_labels) > 0:
                correct = (flat_preds == flat_labels).sum()
                acc = float(correct) / len(flat_labels)

                from sklearn.metrics import f1_score
                macro_f1 = f1_score(
                    flat_labels, flat_preds,
                    average="macro", zero_division=0
                )
            else:
                macro_f1 = 0.0
                acc = 0.0

            return {
                "macro_f1": macro_f1,
                "accuracy": acc
            }

        import torch
        use_bf16 = self.config.training.mixed_precision == "bf16"
        use_fp16 = self.config.training.mixed_precision == "fp16"
        
        # 런타임에 GPU 및 BF16 지원 여부 확인하여 안전하게 Fallback
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Disabling FP16/BF16 and using CPU.")
            use_bf16 = False
            use_fp16 = False
        elif use_bf16 and not torch.cuda.is_bf16_supported():
            logger.warning("BF16 is not supported on this device. Falling back to FP16.")
            use_bf16 = False
            use_fp16 = True

        # Deprecation 경고 대응: warmup_steps 수동 계산
        calc_warmup_steps = int(max_steps * self.config.training.warmup_ratio)

        sft_config = SFTConfig(
            output_dir=self.config.training.output_dir,
            num_train_epochs=self.config.training.num_epochs if max_steps < 0 else 1,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_steps=calc_warmup_steps,
            max_steps=max_steps,
            bf16=use_bf16,
            fp16=use_fp16,
            use_cpu=not torch.cuda.is_available(),
            save_strategy=self.config.training.save_strategy,
            save_steps=self.config.training.save_steps,
            eval_strategy="steps" if val_dataset else "no",
            eval_steps=self.config.training.eval_steps if val_dataset else None,
            save_total_limit=self.config.training.save_total_limit,
            logging_steps=10,
            seed=self.config.training.seed,
            dataloader_pin_memory=True,
            dataloader_drop_last=True,
            load_best_model_at_end=load_best,
            metric_for_best_model=metric_best,
            greater_is_better=greater_better,
            neftune_noise_alpha=self.extra_optimization.get("neftune_noise_alpha", 5), # NEFTune 적용
            packing=False,
            max_length=1024,
        )

        # 모니터링 Callback 및 Early Stopping Callback 정의
        callbacks = [
            RealTimeMonitoringCallback(
                start_time=self.start_time,
                output_dir=Path(self.config.training.output_dir).parent,
                tokenizer=self.tokenizer,
                validation_dataset=val_dataset
            )
        ]
        if load_best and patience > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
            logger.info(f"✓ Early Stopping enabled (patience={patience})")

        # LoRA+ 적용 커스텀 트레이너 인스턴스화
        self.trainer = LoRAPlusSFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=sft_config,
            formatting_func=formatting_func,
            data_collator=data_collator,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )

        logger.info("\nStarting Campaign v2 Training...")
        # Checkpoint Resume 처리
        resume_arg = self.resume_from_checkpoint
        if resume_arg:
            # output_dir에서 최신 체크포인트 확인
            ckpt_dir = Path(self.config.training.output_dir)
            ckpts = list(ckpt_dir.glob("checkpoint-*"))
            if ckpts:
                # 스텝 번호 기준 최신 checkpoint 선택
                ckpts.sort(key=lambda x: int(x.name.split("-")[-1]))
                resume_arg = str(ckpts[-1])
                logger.info(f"Resuming from checkpoint: {resume_arg}")
            else:
                resume_arg = None
                logger.info("No checkpoint found to resume. Starting from scratch.")

        self.trainer.train(resume_from_checkpoint=resume_arg)

        # 최상의 모델 및 토크나이저 저장
        final_save_dir = Path(self.config.training.output_dir).parent / "best_model"
        logger.info(f"Saving best model and tokenizer to {final_save_dir}")
        self.trainer.save_model(str(final_save_dir))
        self.tokenizer.save_pretrained(str(final_save_dir))

        # 메타데이터 기록
        experiment_dir = final_save_dir / "experiment"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "base_model": self.config.model.name,
            "rank": self.rank,
            "total_steps": self.trainer.state.global_step,
            "training_time_seconds": time.time() - self.start_time
        }
        with open(experiment_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info("✓ Training script complete.")


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-1.5B Campaign V2 Fine-Tuning")
    parser.add_argument("--config", type=str, default="configs/qwen15_v2.yaml", help="Configuration file path")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps limit (useful for smoke test)")
    parser.add_argument("--rank", type=int, default=32, help="LoRA Rank to use (16, 32, 64)")
    args = parser.parse_args()

    # 로그 셋업
    setup_logging(log_dir="./logs", level="INFO")
    
    trainer = ModelTrainerV2(
        config_path=args.config,
        resume_from_checkpoint=args.resume,
        max_steps=args.max_steps,
        rank=args.rank
    )
    trainer.train()


if __name__ == "__main__":
    main()
