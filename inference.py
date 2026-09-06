#!/usr/bin/env python3
"""
Inference pipeline with base model + LoRA adapter.

Features:
- Load base model and LoRA adapter
- Grammar-safe JSON generation
- Tool call execution
- Execution time tracking
- GPU memory monitoring
"""

import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import psutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from configs.config import Config
from adapters import get_adapter
from configs.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


class ModelInference:
    """Inference engine for tool calling."""
    
    def __init__(
        self,
        config: Config,
        model_path: str,
        adapter_path: Optional[str] = None
    ):
        """
        Initialize inference engine.
        
        Args:
            config: Configuration object
            model_path: Path to base model
            adapter_path: Path to LoRA adapter (optional)
        """
        self.config = config
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.logger = get_logger(self.__class__.__name__)
        
        # Load adapter for formatting
        try:
            self.adapter = get_adapter(self.config.dataset.adapter)
        except Exception as e:
            self.logger.warning(f"Could not load adapter formatting helper: {str(e)}")
            self.adapter = None
            
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load model and adapter."""
        self.logger.info(f"Loading base model: {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=getattr(torch, self.config.model.torch_dtype),
            device_map=self.config.model.device_map,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        
        # Load adapter if provided
        if self.adapter_path:
            self.logger.info(f"Loading LoRA adapter: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        
        # Set to eval mode
        self.model.eval()
        
        self.logger.info("✓ Model loaded successfully")
    
    @torch.no_grad()
    def generate_tool_call(
        self,
        prompt: str,
        tools: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Generate tool call from prompt.
        
        Args:
            prompt: User prompt
            tools: Available tools
            
        Returns:
            Dict with tool_call, latency, and memory info
        """
        start_time = time.time()
        
        # Format messages using adapter if available, otherwise use fallback
        if self.adapter:
            messages = self.adapter.convert_to_inference_format(prompt, tools)
        else:
            # Build system message with tools
            system_message = "You are a helpful assistant with access to tools. "
            if tools:
                system_message += f"Available tools: {json.dumps(tools)}\n"
                system_message += "Respond with a valid JSON tool call only."
            
            # Format messages
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        
        # Tokenize
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # Generate
        input_ids = inputs["input_ids"]
        
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.config.inference.max_new_tokens,
            temperature=self.config.inference.temperature,
            top_p=self.config.inference.top_p,
            top_k=self.config.inference.top_k,
            do_sample=self.config.inference.do_sample,
            repetition_penalty=self.config.inference.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Decode
        output_text = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        latency = time.time() - start_time
        
        # Parse tool call
        tool_call = self._parse_tool_call(output_text)
        
        # GPU memory
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024  # GB
        
        return {
            "tool_call": tool_call,
            "raw_output": output_text,
            "latency_seconds": latency,
            "gpu_memory_gb": gpu_memory,
            "num_tokens": len(self.tokenizer.encode(output_text))
        }
    
    @staticmethod
    def _parse_tool_call(output: str) -> Dict[str, Any]:
        """
        Parse tool call from output.
        
        Tries JSON parsing, returns error if invalid.
        """
        try:
            # Try to extract JSON
            start_idx = output.find('{')
            end_idx = output.rfind('}')
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = output[start_idx:end_idx + 1]
                tool_call = json.loads(json_str)
                return {"status": "success", "data": tool_call}
            
            return {"status": "error", "message": "No JSON found in output"}
        
        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "message": f"Invalid JSON: {str(e)}",
                "output": output[:200]
            }


class InteractiveInference:
    """Interactive inference shell."""
    
    def __init__(self, inference: ModelInference, tools: Optional[list] = None):
        """Initialize interactive shell."""
        self.inference = inference
        self.tools = tools or []
        self.logger = get_logger(self.__class__.__name__)
    
    def run(self) -> None:
        """Run interactive shell."""
        self.logger.info("=" * 80)
        self.logger.info("INTERACTIVE INFERENCE")
        self.logger.info("=" * 80)
        self.logger.info("Type 'exit' to quit, 'help' for commands\n")
        
        while True:
            try:
                prompt = input("User: ").strip()
                
                if prompt.lower() == "exit":
                    self.logger.info("Goodbye!")
                    break
                
                if prompt.lower() == "help":
                    self._show_help()
                    continue
                
                if not prompt:
                    continue
                
                # Generate tool call
                result = self.inference.generate_tool_call(prompt, self.tools)
                
                # Display results
                self._display_result(result)
                
            except KeyboardInterrupt:
                self.logger.info("\nGoodbye!")
                break
    
    def _display_result(self, result: Dict[str, Any]) -> None:
        """Display inference result."""
        print("\n" + "=" * 80)
        print("Tool Call Result:")
        print("=" * 80)
        
        tool_call = result.get("tool_call", {})
        if tool_call.get("status") == "success":
            print(json.dumps(tool_call.get("data", {}), indent=2, ensure_ascii=False))
        else:
            print(f"Error: {tool_call.get('message', 'Unknown error')}")
        
        print(f"\nMetrics:")
        print(f"  Latency: {result['latency_seconds']:.2f}s")
        print(f"  GPU Memory: {result['gpu_memory_gb']:.2f} GB")
        print(f"  Output Tokens: {result['num_tokens']}")
        print("=" * 80 + "\n")
    
    def _show_help(self) -> None:
        """Show help message."""
        print("\nCommands:")
        print("  exit - Exit interactive shell")
        print("  help - Show this message")
        print("\nTools Available:")
        for tool in self.tools[:5]:
            print(f"  - {tool.get('name', 'unknown')}")
        if len(self.tools) > 5:
            print(f"  ... and {len(self.tools) - 5} more")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Inference with LoRA adapter")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Config file path"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Base model path or name"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="LoRA adapter path"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to test (if not provided, launches interactive shell)"
    )
    parser.add_argument(
        "--tools",
        type=str,
        default=None,
        help="Tools JSON file path"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_dir="./logs", level="INFO")
    
    # Load config
    config = Config.from_yaml(args.config)
    
    # Load tools if provided
    tools = []
    if args.tools and Path(args.tools).exists():
        with open(args.tools) as f:
            tools = json.load(f)
    
    # Initialize inference
    inference = ModelInference(config, args.model, args.adapter)
    
    # Single prompt or interactive
    if args.prompt:
        result = inference.generate_tool_call(args.prompt, tools)
        print(json.dumps(result, indent=2))
    else:
        interactive = InteractiveInference(inference, tools)
        interactive.run()


if __name__ == "__main__":
    main()
