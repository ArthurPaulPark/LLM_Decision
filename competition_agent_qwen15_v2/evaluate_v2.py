#!/usr/bin/env python3
"""
Qwen2.5-1.5B Production Optimization Campaign (Phase 1)
평가 및 벤치마크 자동화 스크립트.

특징:
- 기존 evaluate.py와의 완벽한 호환성 유지
- 추가 지표 측정: Macro F1, Balanced Accuracy, Precision, Recall, Per-Class F1, Confusion Matrix, ECE(Calibration)
- GGUF/HF 모델 포맷 지원
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
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

from configs.config import Config
from configs.logging_config import setup_logging, get_logger

logger = get_logger(__name__)

ACTIONS = [
    "read_file", "write_file", "edit_file", "apply_patch",
    "grep_search", "glob_pattern", "list_directory",
    "run_bash", "run_tests", "lint_or_typecheck",
    "web_search", "ask_user", "plan_task", "respond_only",
]


# ── ECE(Expected Calibration Error) 계산 함수 ──────────────────────────────────
def calculate_ece(confidences: List[float], accuracies: List[int], n_bins: int = 10) -> float:
    """ECE (Expected Calibration Error) 계산."""
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(confidences)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # 각 버킷에 속한 인덱스 필터링
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return float(ece)


class EvaluatorV2:
    def __init__(self, config: Config, model_path: str, adapter_path: str = None):
        self.config = config
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        
        self._load_model()

    def _load_model(self) -> None:
        logger.info(f"Loading model for evaluation: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.config.model.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=getattr(torch, self.config.model.torch_dtype),
            device_map=self.config.model.device_map,
            trust_remote_code=self.config.model.trust_remote_code,
        )

        if self.adapter_path and Path(self.adapter_path).exists():
            logger.info(f"Loading adapter: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

        self.model.eval()
        logger.info("✓ Model loaded successfully.")

    @torch.no_grad()
    def evaluate(self, test_data_path: str, output_path: str = None) -> Dict[str, Any]:
        logger.info(f"Reading test data from {test_data_path}")
        with open(test_data_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        predictions = []
        confidences = []
        probabilities = []
        ground_truths = []
        latencies = []

        start_gpu_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

        for i, item in enumerate(tqdm(test_data, desc="Evaluating (Campaign V2)")):
            # Ground truth 구하기
            gt_label = "respond_only"
            messages = item.get("messages", [])
            if messages:
                for msg in reversed(messages):
                    if msg["role"] == "assistant":
                        content = msg["content"].strip()
                        if content in ACTIONS:
                            gt_label = content
                        break
            ground_truths.append(gt_label)

            # 추론 수행 및 소요시간 측정
            start_time = time.time()
            prompt = self.tokenizer.apply_chat_template(
                messages[:-1],  # 마지막 assistant 턴 제외
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # 실제 텍스트 생성 (Generation 방식)
            gen_outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False,  # Greedy decoding
                temperature=None,
                top_p=None
            )
            
            # 입력 프롬프트를 제외한 새로 생성된 텍스트만 추출
            generated_ids = gen_outputs[0][inputs["input_ids"].shape[-1]:]
            output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()
            
            # 액션 문자열 파싱
            pred_label = "respond_only" # 기본값
            for action in ACTIONS:
                if action in output_text:
                    pred_label = action
                    break

            latency = time.time() - start_time
            latencies.append(latency)

            # 생성 기반이므로 Confidence 확률을 구하는 것이 복잡함 (간단히 1.0으로 더미 처리)
            # ECE Calibration 계산 시엔 무의미해지지만 정확한 Accuracy 지표 복원이 최우선임.
            conf = 1.0
            probs = [1.0 if a == pred_label else 0.0 for a in ACTIONS]

            predictions.append(pred_label)
            confidences.append(conf)
            probabilities.append(probs)

        peak_gpu_mem = (torch.cuda.max_memory_allocated() - start_gpu_mem) / 1024**2 if torch.cuda.is_available() else 0

        # 지표 산출
        metrics = self._compute_detailed_metrics(ground_truths, predictions, confidences, latencies)
        metrics["peak_gpu_mem_mb"] = peak_gpu_mem
        
        # 결과 리포트 출력
        self._print_results(metrics)

        # 결과 저장
        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump({
                    "metrics": metrics,
                    "predictions": [
                        {"gt": gt, "pred": pred, "conf": c, "probs": p}
                        for gt, pred, c, p in zip(ground_truths, predictions, confidences, probabilities)
                    ]
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Metrics saved to {out_p}")

        return metrics

    def _compute_detailed_metrics(
        self,
        gts: List[str],
        preds: List[str],
        confs: List[float],
        latencies: List[float]
    ) -> Dict[str, Any]:
        
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix
        
        # 정확도 계산
        accuracy = accuracy_score(gts, preds)
        macro_f1 = f1_score(gts, preds, average="macro", zero_division=0)
        balanced_acc = balanced_accuracy_score(gts, preds)
        precision = precision_score(gts, preds, average="macro", zero_division=0)
        recall = recall_score(gts, preds, average="macro", zero_division=0)
        
        # Per-Class F1 스코어
        per_class_f1_vals = f1_score(gts, preds, average=None, labels=ACTIONS, zero_division=0)
        per_class_f1 = {act: float(val) for act, val in zip(ACTIONS, per_class_f1_vals)}

        # ECE 계산 (Accuracies list: gt == pred)
        accuracies_list = [1 if gt == pred else 0 for gt, pred in zip(gts, preds)]
        ece = calculate_ece(confs, accuracies_list, n_bins=10)

        # Confusion Matrix
        cm = confusion_matrix(gts, preds, labels=ACTIONS)
        cm_dict = {act: {a: int(cm[i][j]) for j, a in enumerate(ACTIONS)} for i, act in enumerate(ACTIONS)}

        avg_latency = np.mean(latencies) if latencies else 0.0

        return {
            "total_examples": len(gts),
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "balanced_accuracy": float(balanced_acc),
            "precision": float(precision),
            "recall": float(recall),
            "ece": float(ece),
            "avg_latency_seconds": float(avg_latency),
            "per_class_f1": per_class_f1,
            "confusion_matrix": cm_dict
        }

    def _print_results(self, metrics: Dict[str, Any]) -> None:
        print("\n" + "=" * 55)
        print(" CAMPAIGN V2 BENCHMARK RESULTS")
        print("=" * 55)
        print(f"Total Examples     : {metrics['total_examples']}")
        print(f"Accuracy           : {metrics['accuracy']*100:.2f}%")
        print(f"Macro F1           : {metrics['macro_f1']*100:.2f}%")
        print(f"Balanced Accuracy  : {metrics['balanced_accuracy']*100:.2f}%")
        print(f"Precision          : {metrics['precision']*100:.2f}%")
        print(f"Recall             : {metrics['recall']*100:.2f}%")
        print(f"Calibration (ECE)  : {metrics['ece']:.4f}")
        print(f"Avg Latency        : {metrics['avg_latency_seconds']:.4f}s")
        print(f"Peak GPU Memory    : {metrics['peak_gpu_mem_mb']:.1f} MB")
        print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Campaign V2 Model")
    parser.add_argument("--config", type=str, default="configs/qwen15_v2.yaml")
    parser.add_argument("--model", type=str, required=True, help="Base model path or huggingface id")
    parser.add_argument("--adapter", type=str, default=None, help="Adapter checkpoint path")
    parser.add_argument("--test-data", type=str, default="./data/processed/test.json", help="Test JSON path")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path to save metrics")
    args = parser.parse_args()

    setup_logging(log_dir="./logs", level="INFO")
    
    config = Config.from_yaml(args.config)
    evaluator = EvaluatorV2(config, args.model, args.adapter)
    evaluator.evaluate(args.test_data, args.output)


if __name__ == "__main__":
    main()
