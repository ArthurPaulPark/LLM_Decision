#!/usr/bin/env python3
"""
Evaluation script for function calling accuracy.

Metrics:
- Function Accuracy
- Argument Accuracy
- Tool Call Accuracy
- Exact Match
- JSON Valid Rate
- Average Latency
- Peak GPU Memory
- Inference Speed (tokens/sec)
"""

import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

from configs.config import Config
from configs.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


class Evaluator:
    """Evaluation engine for function calling models."""
    
    def __init__(
        self,
        config: Config,
        model_path: str,
        adapter_path: str
    ):
        """Initialize evaluator."""
        self.config = config
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.logger = get_logger(self.__class__.__name__)
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load model and adapter."""
        self.logger.info(f"Loading model: {self.model_path}")
        
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
            self.logger.info(f"Loading adapter: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        
        self.model.eval()
        self.logger.info("✓ Model loaded")
    
    def evaluate(self, test_data_path: str) -> Dict[str, Any]:
        """
        Evaluate model on test dataset.
        
        Args:
            test_data_path: Path to test JSON file
            
        Returns:
            Evaluation metrics dictionary
        """
        self.logger.info("=" * 80)
        self.logger.info("EVALUATION PIPELINE")
        self.logger.info("=" * 80)
        
        # Load test data
        self.logger.info(f"\nLoading test data from {test_data_path}")
        with open(test_data_path) as f:
            test_data = json.load(f)
        
        self.logger.info(f"✓ Loaded {len(test_data)} examples")
        
        # Evaluate
        predictions = []
        latencies = []
        
        self.logger.info("\nGenerating predictions...")
        for item in tqdm(test_data, desc="Evaluating"):
            pred = self._predict_single(item)
            predictions.append(pred)
            if "latency" in pred:
                latencies.append(pred["latency"])
        
        # Compute metrics
        self.logger.info("\nComputing metrics...")
        metrics = self._compute_metrics(test_data, predictions)
        metrics["latencies"] = latencies
        
        # Print results
        self._print_results(metrics)
        
        # Save report
        self._save_report(metrics, predictions)
        
        return metrics
    
    @torch.no_grad()
    def _predict_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prediction for single example."""
        try:
            start_time = time.time()
            
            # Extract ground truth tool call
            messages = item.get("messages", [])
            ground_truth = None
            if messages and len(messages) >= 3:
                ground_truth = messages[-1].get("content", {})
                if isinstance(ground_truth, str):
                    try:
                        ground_truth = json.loads(ground_truth)
                    except:
                        ground_truth = None
            
            if not messages:
                return {"error": "No messages found in item"}
            
            # Extract system and user messages (exclude final assistant response)
            input_messages = messages[:-1]
            
            # Format using tokenizer chat template (exactly matches training format)
            text = self.tokenizer.apply_chat_template(
                input_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            output_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            latency = time.time() - start_time
            
            # Parse prediction
            predicted_call = self._parse_tool_call(output_text)
            
            return {
                "ground_truth": ground_truth,
                "predicted": predicted_call,
                "raw_output": output_text,
                "latency": latency,
                "valid_json": predicted_call.get("status") == "success"
            }
        
        except Exception as e:
            return {"error": str(e)}
    
    def _compute_metrics(
        self,
        test_data: List[Dict],
        predictions: List[Dict]
    ) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        metrics = {
            "total": len(predictions),
            "valid_json": 0,
            "exact_match": 0,
            "tool_accuracy": 0,
            "argument_accuracy": 0,
            "function_accuracy": 0,
            "errors": 0,
            "tool_stats": defaultdict(lambda: {"total": 0, "correct": 0})
        }
        
        argument_scores = []
        
        for pred in predictions:
            if "error" in pred:
                metrics["errors"] += 1
                continue
            
            if not pred.get("valid_json"):
                continue
            
            metrics["valid_json"] += 1
            
            ground_truth = pred.get("ground_truth")
            predicted = pred.get("predicted", {}).get("data", {})
            
            if not ground_truth or not predicted:
                continue
            
            # Tool name accuracy
            gt_tool = ground_truth.get("tool_name") or ground_truth.get("function", "")
            pred_tool = predicted.get("tool_name") or predicted.get("function", "")
            
            if gt_tool == pred_tool:
                metrics["tool_accuracy"] += 1
                metrics["tool_stats"][gt_tool]["correct"] += 1
            
            metrics["tool_stats"][gt_tool]["total"] += 1
            
            # Arguments accuracy
            gt_args = ground_truth.get("arguments", {})
            pred_args = predicted.get("arguments", {})
            
            if gt_args == pred_args:
                metrics["argument_accuracy"] += 1
            elif isinstance(gt_args, dict) and isinstance(pred_args, dict):
                matching_args = sum(
                    1 for k in gt_args if k in pred_args and gt_args[k] == pred_args[k]
                )
                if gt_args:
                    arg_score = matching_args / len(gt_args)
                    argument_scores.append(arg_score)
            
            # Exact match
            if ground_truth == predicted:
                metrics["exact_match"] += 1
        
        # Normalize accuracies
        if metrics["valid_json"] > 0:
            metrics["tool_accuracy"] = metrics["tool_accuracy"] / metrics["valid_json"]
            metrics["argument_accuracy"] = metrics["argument_accuracy"] / metrics["valid_json"]
            metrics["function_accuracy"] = metrics["tool_accuracy"]
        
        if metrics["total"] > 0:
            metrics["exact_match"] = metrics["exact_match"] / metrics["total"]
            metrics["json_valid_rate"] = metrics["valid_json"] / metrics["total"]
            metrics["error_rate"] = metrics["errors"] / metrics["total"]
        
        metrics["avg_argument_score"] = statistics.mean(argument_scores) if argument_scores else 0.0
        
        return metrics
    
    @staticmethod
    def _parse_tool_call(output: str) -> Dict[str, Any]:
        """Parse tool call from output."""
        try:
            start_idx = output.find('{')
            end_idx = output.rfind('}')
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = output[start_idx:end_idx + 1]
                data = json.loads(json_str)
                return {"status": "success", "data": data}
            
            return {"status": "error", "message": "No JSON found"}
        
        except json.JSONDecodeError:
            return {"status": "error", "message": "Invalid JSON"}
    
    def _print_results(self, metrics: Dict[str, Any]) -> None:
        """Print evaluation results."""
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        
        print(f"\nDataset Statistics:")
        print(f"  Total Examples: {metrics['total']}")
        print(f"  Valid JSON: {metrics['valid_json']} ({metrics.get('json_valid_rate', 0)*100:.1f}%)")
        print(f"  Errors: {metrics['errors']} ({metrics.get('error_rate', 0)*100:.1f}%)")
        
        print(f"\nAccuracy Metrics:")
        print(f"  Function/Tool Accuracy: {metrics['function_accuracy']*100:.2f}%")
        print(f"  Argument Accuracy: {metrics['argument_accuracy']*100:.2f}%")
        print(f"  Exact Match: {metrics['exact_match']*100:.2f}%")
        print(f"  Avg Argument Score: {metrics.get('avg_argument_score', 0)*100:.2f}%")
        
        if metrics.get("latencies"):
            lat = metrics["latencies"]
            print(f"\nLatency Metrics:")
            print(f"  Min: {min(lat):.2f}s")
            print(f"  Max: {max(lat):.2f}s")
            print(f"  Mean: {statistics.mean(lat):.2f}s")
            print(f"  Median: {statistics.median(lat):.2f}s")
        
        print("\n" + "=" * 80)
    
    def _save_report(self, metrics: Dict[str, Any], predictions: List[Dict]) -> None:
        """Save evaluation report."""
        report_dir = Path(self.config.paths.outputs_dir) / "evaluation"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_file = report_dir / "metrics.json"
        metrics_to_save = {k: v for k, v in metrics.items() if k != "latencies"}
        with open(metrics_file, "w") as f:
            json.dump(metrics_to_save, f, indent=2, default=str)
        
        # Save predictions
        pred_file = report_dir / "predictions.json"
        with open(pred_file, "w") as f:
            json.dump(predictions[:100], f, indent=2)  # Save first 100
        
        self.logger.info(f"✓ Report saved to {report_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--test-data", type=str, default="./data/processed/test.json")
    
    args = parser.parse_args()
    
    setup_logging(log_dir="./logs", level="INFO")
    
    config = Config.from_yaml(args.config)
    config.create_directories()
    
    evaluator = Evaluator(config, args.model, args.adapter)
    metrics = evaluator.evaluate(args.test_data)


if __name__ == "__main__":
    main()
