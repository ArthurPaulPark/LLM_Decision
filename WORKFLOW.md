# LLM Decision Framework - Complete Workflow Guide

## Overview

This guide walks you through the complete workflow of the LLM Decision Framework, from preprocessing to inference to evaluation.

## Prerequisites

- Python 3.11+
- NVIDIA GPU with 16GB VRAM (T4 recommended)
- CUDA Toolkit 11.8+ (for GPU support)
- Git

## 1. Environment Setup

### Installation

```bash
# Clone repository
git clone <repository-url>
cd llm-decision-framework

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings (optional)
# HUGGINGFACE_TOKEN=your_token
# WANDB_ENABLED=true
```

## 2. Data Preprocessing

### Step 1: Download and Analyze Dataset

```bash
python preprocess.py --config configs/base.yaml
```

This will:
- ✅ Download Salesforce xLAM dataset from HuggingFace
- ✅ Parse tool schemas
- ✅ Analyze dataset structure
- ✅ Print statistics to console
- ✅ Save to `./logs/experiment_*.log`

**Expected Output:**
```
========================================
PREPROCESSING PIPELINE
========================================

[1/5] Downloading dataset...
✓ Downloaded successfully. Splits: dict_keys(['train', 'validation', 'test'])

[2/5] Loading tool schemas...
✓ Loaded 128 unique tools

[3/5] Loading and analyzing examples...
✓ Loaded 40000 examples from train
✓ Loaded 5000 examples from validation
✓ Loaded 10000 examples from test

[4/5] Converting to chat format...
Converting train: 100%|████████| 40000/40000 [2:14<00:00, 297.4ex/s]
✓ Saved train (40000 examples) to ./data/processed/train.json

Split Distribution:
  train        : 40000 (80.0%)
  validation   :  5000 (10.0%)
  test         : 10000 (10.0%)

Total Unique Tools: 128
```

### Step 2: Verify Processed Data

```bash
# Check processed files
ls -lh data/processed/

# Sample processing output
cat data/processed/train.json | head -c 500
```

**Files Created:**
- `data/processed/train.json` - Training examples in chat format
- `data/processed/validation.json` - Validation examples
- `data/processed/test.json` - Test examples
- `data/processed/tools.json` - Tool schema reference

## 3. Model Training

### Option A: LoRA Training (Recommended for T4)

```bash
# LoRA configuration optimized for 16GB VRAM
python train.py --config configs/lora.yaml
```

**Configuration:**
- Batch size: 4
- Learning rate: 5e-4
- Epochs: 3
- LoRA rank: 8
- Mixed precision: BF16

### Option B: QLoRA Training (More Memory Efficient)

```bash
# QLoRA with 4-bit quantization
python train.py --config configs/qlora.yaml
```

**Configuration:**
- Batch size: 2 (more memory efficient)
- 4-bit quantization (NF4)
- Same LoRA rank: 8

### Training Output

```
========================================
TRAINING PIPELINE
========================================

[1/3] Preparing model...
Loading model: Qwen/Qwen2.5-0.5B
✓ Tokenizer loaded (vocab size: 151936)
✓ Model prepared with LORA

Trainable params: 5,898,240 / 487,158,784 (1.21%)

[2/3] Preparing datasets...
Loading datasets...
✓ Loaded 40000 training examples
✓ Loaded 5000 validation examples

[3/3] Setting up training...

Starting training...
Epoch 1/3: 100%|████████| 10000/10000 [1:15:30<00:00, 0.45 ex/s]
Loss: 0.2314 | Eval Loss: 0.2108

Epoch 2/3: 100%|████████| 10000/10000 [1:15:25<00:00, 0.45 ex/s]
Loss: 0.1834 | Eval Loss: 0.1956

Epoch 3/3: 100%|████████| 10000/10000 [1:15:28<00:00, 0.45 ex/s]
Loss: 0.1623 | Eval Loss: 0.1845

✓ TRAINING COMPLETED
✓ Experiment saved to outputs/checkpoints/experiment
```

### Monitor Training

In another terminal:

```bash
# Monitor with TensorBoard
tensorboard --logdir outputs/tensorboard

# Open browser: http://localhost:6006
```

### Resume Training

If training was interrupted:

```bash
python train.py --config configs/lora.yaml --resume outputs/checkpoints/checkpoint-2000
```

## 4. Model Inference

### Interactive Shell

```bash
python inference.py \
    --model "Qwen/Qwen2.5-0.5B" \
    --adapter "outputs/checkpoints/adapter_model" \
    --tools "data/processed/tools.json"
```

**Interactive Commands:**
```
User: Schedule a meeting tomorrow at 2 PM
================================================================================
Tool Call Result:
================================================================================
{
  "tool_name": "schedule_meeting",
  "arguments": {
    "time": "2 PM",
    "date": "tomorrow"
  }
}

Metrics:
  Latency: 0.32s
  GPU Memory: 7.2 GB
  Output Tokens: 45
================================================================================

User: exit
Goodbye!
```

### Single Prompt

```bash
python inference.py \
    --model "Qwen/Qwen2.5-0.5B" \
    --adapter "outputs/checkpoints/adapter_model" \
    --prompt "Send an email to john@example.com" \
    --tools "data/processed/tools.json"
```

**Output:**
```json
{
  "tool_call": {
    "status": "success",
    "data": {
      "tool_name": "send_email",
      "arguments": {
        "recipient": "john@example.com"
      }
    }
  },
  "latency_seconds": 0.28,
  "gpu_memory_gb": 7.2,
  "num_tokens": 42
}
```

## 5. Model Evaluation

### Run Evaluation

```bash
python evaluate.py \
    --config configs/base.yaml \
    --model "Qwen/Qwen2.5-0.5B" \
    --adapter "outputs/checkpoints/adapter_model" \
    --test-data "data/processed/test.json"
```

**Evaluation Output:**
```
================================================================================
EVALUATION RESULTS
================================================================================

Dataset Statistics:
  Total Examples: 10000
  Valid JSON: 9950 (99.5%)
  Errors: 50 (0.5%)

Accuracy Metrics:
  Function/Tool Accuracy: 94.25%
  Argument Accuracy: 91.83%
  Exact Match: 89.42%
  Avg Argument Score: 93.12%

Latency Metrics:
  Min: 0.24s
  Max: 0.45s
  Mean: 0.32s
  Median: 0.30s

✓ Report saved to outputs/evaluation
```

### Review Results

```bash
# View metrics
cat outputs/evaluation/metrics.json | python -m json.tool

# Sample predictions
cat outputs/evaluation/predictions.json | head -c 1000
```

## 6. Advanced: Adding New Dataset

### Create New Adapter

```python
# adapters/new_dataset_adapter.py
from adapters.base_adapter import DatasetAdapter, ToolSchema, Example
from typing import List, Dict, Any

class NewDatasetAdapter(DatasetAdapter):
    def __init__(self):
        super().__init__(dataset_name="company/new-dataset", version="1.0")
    
    def download(self, cache_dir: str) -> None:
        # Implement download logic
        pass
    
    def load_tools(self) -> List[ToolSchema]:
        # Load and parse tools
        pass
    
    def load_examples(self, split: str = "train") -> List[Example]:
        # Load examples from your dataset
        pass
    
    def convert_to_chat_format(self, example: Example, tools: List[ToolSchema]) -> Dict[str, Any]:
        # Convert to Qwen chat format
        pass
    
    def parse_prediction(self, output: str, tools: List[ToolSchema]) -> Dict[str, Any]:
        # Parse model output
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        # Return dataset statistics
        pass
    
    def get_split_info(self) -> Dict[str, int]:
        # Return split information
        pass
```

### Register Adapter

```python
# adapters/__init__.py
from adapters.base_adapter import DatasetAdapter
from adapters.xlam_adapter import XLAMAdapter
from adapters.new_dataset_adapter import NewDatasetAdapter

__all__ = ["DatasetAdapter", "XLAMAdapter", "NewDatasetAdapter"]
```

### Update Configuration

```yaml
# configs/base.yaml
dataset:
  name: "company/new-dataset"
  adapter: "new_dataset"  # lowercase filename without .py
```

### Run Pipeline

```bash
# Everything else remains unchanged!
python preprocess.py --config configs/base.yaml
python train.py --config configs/lora.yaml
python evaluate.py --model MODEL_PATH --adapter ADAPTER_PATH
```

## 7. Troubleshooting

### CUDA Out of Memory

```bash
# Use QLoRA instead of LoRA
python train.py --config configs/qlora.yaml

# Or reduce batch size in config file
```

### Model Download Issues

```bash
# Manually download model
huggingface-cli download Qwen/Qwen2.5-0.5B

# Or set cache directory
export HF_HOME=/custom/cache/path
```

### Slow Training

- Use `torch.compile()` (requires PyTorch 2.0+)
- Enable faster attention implementations
- Use smaller batch size with gradient accumulation
- Check GPU utilization: `nvidia-smi`

## 8. Production Deployment

### Export Merged Model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

# Load adapter
model = PeftModel.from_pretrained(model, "outputs/checkpoints/adapter_model")

# Merge and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained("models/merged-model")
```

### Docker Support

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3.11 python3-pip
RUN python3 -m pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python3", "inference.py", "--model", "MODEL_PATH", "--adapter", "ADAPTER_PATH"]
```

## Support

For issues or questions, please refer to:
- README.md for overview
- Each script's `--help` flag
- GitHub issues
- Discussions forum

Happy training! 🚀
