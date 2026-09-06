#!/usr/bin/env python3
import sys
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
"""
V3 Hard Sample Dataset Engineering
- 기존 V2 모델로 train.json의 일부분을 평가하여, 틀린 문제(Hard Samples)만 추출합니다.
- 배치 추론(Batch Inference) 적용.
"""

import json
import time
import random
import logging
import argparse
import sys
from pathlib import Path

# Mock BloomPreTrainedModel to prevent ImportError in older versions of peft
import transformers
if not hasattr(transformers, "BloomPreTrainedModel"):
    transformers.BloomPreTrainedModel = type("BloomPreTrainedModel", (object,), {})

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ACTIONS = [
    "read_file", "write_file", "edit_file", "apply_patch",
    "grep_search", "glob_pattern", "list_directory",
    "run_bash", "run_tests", "lint_or_typecheck",
    "web_search", "ask_user", "plan_task", "respond_only",
]

def load_model(base_model: str, adapter_path: str):
    logger.info(f"Loading tokenizer from {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    # 배치 생성을 위한 Left Padding 설정
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading base model (bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path and Path(adapter_path).exists():
        logger.info(f"Loading adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    model.eval()
    return model, tokenizer

def create_hard_samples(model, tokenizer, train_path: str, output_path: str, sample_size: int, batch_size: int):
    logger.info(f"Loading training data from {train_path}...")
    with open(train_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)
    
    logger.info(f"Total samples available: {len(full_data)}")
    
    # 셔플 후 샘플 추출
    random.seed(42)
    random.shuffle(full_data)
    eval_data = full_data[:sample_size]
    logger.info(f"Selected {len(eval_data)} samples for Hard Sample mining.")

    hard_samples = []
    correct_count = 0

    # Batch Processing
    for i in tqdm(range(0, len(eval_data), batch_size), desc="Inferencing"):
        batch = eval_data[i:i + batch_size]
        
        prompts = []
        ground_truths = []
        
        for item in batch:
            messages = item.get("messages", [])
            gt_label = "respond_only"
            # Ground Truth 추출
            if messages:
                for msg in reversed(messages):
                    if msg["role"] == "assistant":
                        content = msg["content"].strip()
                        if content in ACTIONS:
                            gt_label = content
                        break
            ground_truths.append(gt_label)
            
            # 마지막 assistant 답변을 제외한 프롬프트 구성
            prompt = tokenizer.apply_chat_template(
                messages[:-1], 
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)

        # 토크나이징 (Left Padding 적용됨)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.no_grad():
            gen_outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        
        # 생성된 토큰 텍스트 파싱
        generated_ids = gen_outputs[:, input_len:]
        output_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        for idx, (gt, text) in enumerate(zip(ground_truths, output_texts)):
            text_lower = text.strip().lower()
            pred_label = "respond_only"
            for action in ACTIONS:
                if action in text_lower:
                    pred_label = action
                    break
            
            if pred_label == gt:
                correct_count += 1
            else:
                # 틀린 문제(Hard Sample) 추가
                hard_samples.append(batch[idx])

    acc = correct_count / len(eval_data)
    logger.info(f"Mining Accuracy: {acc*100:.2f}% ({correct_count}/{len(eval_data)})")
    logger.info(f"Found {len(hard_samples)} Hard Samples!")

    # JSON 저장
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(hard_samples, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Hard dataset saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", type=str, default="outputs/qwen15_v2/best_model")
    parser.add_argument("--train_data", type=str, default="data/processed/train.json")
    parser.add_argument("--output", type=str, default="data/processed/hard_train.json")
    parser.add_argument("--sample_size", type=int, default=15000, help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    model, tokenizer = load_model(args.base_model, args.adapter)
    create_hard_samples(model, tokenizer, args.train_data, args.output, args.sample_size, args.batch_size)

if __name__ == "__main__":
    main()
