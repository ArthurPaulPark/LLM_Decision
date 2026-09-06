# LLM Decision Framework

A production-quality, reusable, and extensible framework for fine-tuning large language models on function calling tasks. Designed for multi-year AI competitions, this framework supports multiple datasets and models while requiring only adapter changes for new datasets.

## 🎯 Mission

Build a universal framework that works with ANY function calling dataset, not just Salesforce xLAM. When a new competition starts with a different dataset, **only the DatasetAdapter needs to change**. Everything else remains unchanged.

## ⭐ Key Features

- **🔄 Universal Architecture**: Adapter-based design for multiple datasets
- **⚡ Optimized for T4**: Specifically tuned for 16GB VRAM NVIDIA T4 GPU
- **🎛️ LoRA & QLoRA**: Memory-efficient fine-tuning with 4-bit quantization support
- **📊 Comprehensive Evaluation**: Function accuracy, argument accuracy, exact match, JSON validity
- **🎨 Modern Dashboard**: Streamlit + FastAPI for real-time monitoring and management
- **📈 Production Ready**: Type hints, logging, configuration management, error handling
- **🔬 Research Grade**: Clean separation of concerns, extensible architecture

## 📋 Project Structure

```
llm-decision-framework/
├── adapters/                          # Dataset adapters (only change for new datasets)
│   ├── base_adapter.py               # Abstract adapter interface
│   ├── xlam_adapter.py               # xLAM dataset adapter
│   └── __init__.py
├── app/                              # Streamlit frontend
│   ├── dashboard.py                  # Main dashboard
│   └── pages/                        # Additional dashboard pages
├── backend/                          # FastAPI backend
│   └── api.py                        # REST API endpoints
├── configs/                          # Configuration management
│   ├── base.yaml                     # Base configuration
│   ├── lora.yaml                     # LoRA settings (T4 optimized)
│   ├── qlora.yaml                    # QLoRA settings (4-bit)
│   ├── config.py                     # Config dataclasses
│   └── logging_config.py             # Logging setup
├── data/                             # Data directory
│   ├── raw/                          # Raw dataset files
│   ├── processed/                    # Processed dataset (chat format)
│   └── cache/                        # HuggingFace cache
├── models/                           # Downloaded models
├── outputs/                          # Training outputs
│   ├── checkpoints/                  # Model checkpoints
│   ├── predictions/                  # Inference predictions
│   ├── adapters/                     # Saved LoRA adapters
│   ├── tensorboard/                  # TensorBoard logs
│   └── evaluation/                   # Evaluation reports
├── logs/                             # Training logs
├── preprocess.py                     # Data preprocessing
├── train.py                          # Training script
├── evaluate.py                       # Evaluation script
├── inference.py                      # Inference script
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository>
cd llm-decision-framework

# Install dependencies (Python 3.11+)
pip install -r requirements.txt

# For CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Preprocessing

```bash
# Download and preprocess Salesforce xLAM dataset
python preprocess.py --config configs/base.yaml

# This will:
# - Download dataset from HuggingFace
# - Analyze dataset structure
# - Print statistics
# - Convert to Qwen chat format
# - Save processed files
```

### 3. Training

```bash
# LoRA training (T4 optimized)
python train.py --config configs/lora.yaml

# Or QLoRA for more memory efficiency
python train.py --config configs/qlora.yaml

# Resume from checkpoint
python train.py --config configs/lora.yaml --resume ./outputs/checkpoints/checkpoint-1000

# Monitor with TensorBoard
tensorboard --logdir ./outputs/tensorboard
```

### 4. Inference

```bash
# Interactive inference
python inference.py \
    --model "Qwen/Qwen2.5-0.5B" \
    --adapter "./outputs/checkpoints/adapter_model"

# Single prompt
python inference.py \
    --model "Qwen/Qwen2.5-0.5B" \
    --adapter "./outputs/checkpoints/adapter_model" \
    --prompt "Schedule a meeting for tomorrow at 2 PM"

# Load tools
python inference.py \
    --model "Qwen/Qwen2.5-0.5B" \
    --adapter "./outputs/checkpoints/adapter_model" \
    --tools "./data/processed/tools.json"
```

### 5. Evaluation

```bash
# Evaluate on test set
python evaluate.py \
    --model "Qwen/Qwen2.5-0.5B" \
    --adapter "./outputs/checkpoints/adapter_model" \
    --test-data "./data/processed/test.json"

# Generates:
# - metrics.json (accuracy, latency, etc.)
# - predictions.json (sample predictions)
# - evaluation/report.html (visual report)
```

### 6. Dashboard

```bash
# Terminal 1: Start FastAPI backend
python backend/api.py

# Terminal 2: Start Streamlit frontend
streamlit run app/dashboard.py

# Open browser: http://localhost:8501
```

## 📊 Dashboard Pages

### 🏠 Home
- Project overview
- Current configuration
- Quick start guide

### 📁 Dataset
- Dataset statistics and distribution
- Split information
- Dataset preview and samples
- Tool schema visualization

### ⚙️ Training
- Model selection (Qwen, etc.)
- Optimization type (LoRA/QLoRA)
- Hyperparameter configuration
- Training control (start/pause/resume/stop)

### 📈 Monitoring
- Real-time training curves
- Learning rate schedule
- GPU/CPU resource usage
- ETA and training speed
- TensorBoard integration

### 🧪 Inference
- Prompt input interface
- Tool call generation
- Pretty JSON output
- Latency and token statistics

### 📊 Evaluation
- Function accuracy
- Argument accuracy
- JSON validity rate
- Exact match percentage
- Error analysis
- Confusion matrix for tools

### 📂 Models
- Installed models
- Downloaded LoRA adapters
- Merge adapters
- Export/backup functionality

## ⚙️ Configuration

All hyperparameters are configured via YAML files. No magic numbers in code!

### Base Configuration (`configs/base.yaml`)

```yaml
dataset:
  name: "Salesforce/xlam-function-calling-60k"
  adapter: "xlam"

model:
  name: "Qwen/Qwen2.5-0.5B"

training:
  num_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 5.0e-4
  # ... more settings

optimization:
  type: "lora"
  lora_rank: 8
  lora_alpha: 16
  # ... more settings
```

### LoRA Configuration (`configs/lora.yaml`)

Optimized for T4 16GB VRAM:
- Rank: 8
- Batch size: 4
- Gradient accumulation: 2
- BF16 mixed precision

### QLoRA Configuration (`configs/qlora.yaml`)

Ultra memory-efficient:
- 4-bit quantization (NF4)
- Rank: 8
- Batch size: 2
- Gradient accumulation: 4

## 🔄 Dataset Adaptation

### For Salesforce xLAM (Current)

Adapter: `adapters/xlam_adapter.py`

```python
from adapters import XLAMAdapter

adapter = XLAMAdapter()
adapter.download("./data/cache")
tools = adapter.load_tools()
examples = adapter.load_examples("train")
```

### For New Datasets

1. **Subclass `DatasetAdapter`**:
```python
from adapters.base_adapter import DatasetAdapter, ToolSchema, Example

class NewDatasetAdapter(DatasetAdapter):
    def download(self, cache_dir: str) -> None:
        # Implement download logic
        pass
    
    def load_tools(self) -> List[ToolSchema]:
        # Extract tool schemas
        pass
    
    def load_examples(self, split: str) -> List[Example]:
        # Load examples
        pass
    
    # ... implement other abstract methods
```

2. **Update config**:
```yaml
dataset:
  name: "new-dataset/name"
  adapter: "new_adapter"  # lowercase filename
```

3. **Register adapter** in `adapters/__init__.py`:
```python
from adapters.new_adapter import NewDatasetAdapter
```

4. **Run preprocessing**:
```bash
python preprocess.py --config configs/base.yaml
```

That's it! Everything else remains unchanged.

## 🎓 Training Metrics & Monitoring

### TensorBoard Integration

```bash
tensorboard --logdir ./outputs/tensorboard
# View at http://localhost:6006
```

Tracked metrics:
- Loss curve
- Learning rate schedule
- Validation metrics
- Gradient statistics

### Experiment Management

Each training run saves:
- `config.yaml` - Full configuration
- `metadata.json` - Experiment metadata
- `trainer_state.json` - Training state
- Model checkpoint (or LoRA adapter)
- Tokenizer
- Training logs

## 📈 Evaluation Metrics

### Accuracy Metrics

- **Function Accuracy**: % of correct tool names
- **Argument Accuracy**: % of correct argument values
- **Tool Call Accuracy**: Function + Arguments both correct
- **Exact Match**: Entire tool call matches ground truth

### Validity Metrics

- **JSON Valid Rate**: % of valid JSON outputs
- **Format Compliance**: Tool call follows expected schema

### Performance Metrics

- **Average Latency**: Inference time (seconds)
- **Throughput**: Examples per minute
- **GPU Memory**: Peak usage during inference
- **Inference Speed**: Tokens per second

### Error Analysis

- Confusion matrix for tools
- Common argument errors
- Output format violations

## 🔧 Advanced Usage

### Custom Model Architecture

1. Prepare model in HuggingFace format
2. Update `configs/base.yaml`:
```yaml
model:
  name: "your-org/your-model"
  model_type: "base"
```

### Custom Training Strategy

Modify `train.py` to adjust:
- Training loop
- Optimization
- Loss functions
- Evaluation callbacks

### Distributed Training

Use Accelerate for multi-GPU:
```bash
accelerate config
accelerate launch train.py --config configs/lora.yaml
```

## 📝 Type Hints & Code Quality

The entire codebase uses:
- ✅ Type hints (Python 3.11+)
- ✅ Dataclasses for configuration
- ✅ Logging instead of prints
- ✅ PEP 8 compliant
- ✅ SOLID principles
- ✅ Minimal code duplication

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Type checking
mypy preprocess.py train.py inference.py evaluate.py

# Linting
flake8 .
black --check .
```

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Use QLoRA instead
python train.py --config configs/qlora.yaml

# Or reduce batch size in config
```

### Dataset Download Issues

```bash
# Manually set cache directory
python preprocess.py --cache-dir /custom/cache/path

# Check HuggingFace credentials
huggingface-cli login
```

### Model Not Found

```bash
# Download model first
huggingface-cli download Qwen/Qwen2.5-0.5B

# Or use full model path
python train.py --model "/path/to/local/model"
```

### API Connection Error

```bash
# Ensure backend is running
python backend/api.py

# Check if port 8000 is available
lsof -i :8000
```

## ❓ FAQ

### Q: Can I use this with different models?

**A**: Yes! Update `model.name` in config. Any HuggingFace causal language model works.

### Q: How do I adapt this for a new dataset?

**A**: Create a new adapter by subclassing `DatasetAdapter`. Only ~200 lines of code needed. See [Dataset Adaptation](#-dataset-adaptation).

### Q: What's the difference between LoRA and QLoRA?

**A**: 
- **LoRA**: Requires 12GB VRAM, faster inference
- **QLoRA**: Requires 8GB VRAM with 4-bit quantization, slightly slower

### Q: How do I save and export trained adapters?

**A**: 
```bash
# Auto-saved in outputs/checkpoints
# Export for production
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('base_model')
model = PeftModel.from_pretrained(model, 'adapter_path')
model.save_pretrained('exported_adapter')
"
```

### Q: Can I merge adapter with base model?

**A**: 
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("base_model")
model = PeftModel.from_pretrained(model, "adapter")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged_model")
```

### Q: How do I monitor training remotely?

**A**: Use Weights & Biases:
1. `pip install wandb`
2. Enable in `configs/base.yaml`: `wandb_enabled: true`
3. Run: `wandb login`
4. View at https://wandb.ai/

## 📚 Resources

- [🤗 HuggingFace Documentation](https://huggingface.co/docs)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Streamlit Documentation](https://docs.streamlit.io)
- [FastAPI Documentation](https://fastapi.tiangolo.com)

## 📄 License

[Specify your license here]

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Type hints for all functions
- Comprehensive logging
- Configuration via YAML (no hardcoding)
- Tests for new features
- Updated documentation

## 🎓 Citation

If you use this framework, please cite:

```bibtex
@software{llm_decision_2024,
  title={LLM Decision Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/}
}
```

---

**Built with ❤️ for production-grade AI research**

For questions, issues, or feature requests, please open an issue on GitHub.
