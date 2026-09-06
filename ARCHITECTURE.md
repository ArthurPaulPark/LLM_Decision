# Project Architecture & Design

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Entry Points                          │
│         preprocess.py / train.py / inference.py              │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
    ┌─────────┐  ┌──────────┐  ┌──────────┐
    │ preprocess.py │ train.py │ inference.py
    │ evaluate.py  │
    └─────────┐  └──────────┘  └──────────┘
              │
        ┌─────▼──────────────┐
        │ DatasetAdapter     │
        │ (Universal)        │
        └────────┬───────────┘
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
 XLAMAdapter  NewAdapter  FutureAdapter
```

## Core Design Principles

### 1. Adapter Pattern

**Why?** Allow different datasets without changing core code.

```
DatasetAdapter (Abstract)
    ├── XLAMAdapter (Current)
    ├── NewDatasetAdapter (Future)
    └── ...
```

Only adapters change. Everything else stays the same.

### 2. Configuration-Driven

**Why?** No magic numbers. All hyperparameters in YAML.

```yaml
training:
  batch_size: 4        # Change here
  learning_rate: 5e-4  # Not in code
  epochs: 3            # Not in code
```

### 3. Modular Design

**Why?** Clear separation of concerns.

```
Preprocessing   →  Training  →  Evaluation
                      ↓
                   Inference
```

### 4. Type Safety

**Why?** Catch errors early. IDE support.

```python
from configs.config import Config
from adapters import DatasetAdapter

def train(config: Config, adapter: DatasetAdapter) -> None:
    # Type hints for everything
    pass
```

## Module Responsibilities

### Adapters (`adapters/`)

**Purpose**: Bridge between framework and datasets

```
adapter.download()           → Get data from HuggingFace
adapter.load_tools()         → Extract tool schemas
adapter.load_examples()      → Get training examples
adapter.convert_to_chat_format() → Format for model
adapter.parse_prediction()   → Extract tool calls from output
```

### Configs (`configs/`)

**Purpose**: Configuration management

```
base.yaml   → Default settings
lora.yaml   → LoRA optimized
qlora.yaml  → QLoRA optimized
config.py   → Python dataclasses
```

### Preprocessing (`preprocess.py`)

**Purpose**: Data pipeline

1. Download dataset via adapter
2. Parse tool schemas
3. Analyze structure & statistics
4. Convert to chat format
5. Save as JSON

### Training (`train.py`)

**Purpose**: Fine-tuning

1. Load model with LoRA/QLoRA
2. Prepare datasets
3. Create SFT trainer
4. Train with mixed precision
5. Save checkpoints & adapters

### Inference (`inference.py`)

**Purpose**: Tool call generation

1. Load base model + adapter
2. Generate from prompts
3. Parse JSON output
4. Track latency & memory

### Evaluation (`evaluate.py`)

**Purpose**: Accuracy metrics

1. Load test data
2. Generate predictions
3. Compare with ground truth
4. Compute accuracy metrics
5. Generate report

## Data Flow

### Preprocessing Flow

```
HuggingFace
    ↓
[Download] via XLAMAdapter
    ↓
Raw Dataset (train/val/test splits)
    ↓
[Parse Tools] load_tools()
    ↓
ToolSchema[]
    ↓
[Load Examples] load_examples()
    ↓
Example[]
    ↓
[Convert Format] convert_to_chat_format()
    ↓
Chat Messages + Tool Schemas
    ↓
[Save] JSON files
    ↓
data/processed/
    ├── train.json
    ├── validation.json
    ├── test.json
    └── tools.json
```

### Training Flow

```
data/processed/train.json
    ↓
[Load] via transformers.load_dataset()
    ↓
HuggingFace Dataset
    ↓
[Model Preparation]
    ├── Load base model (Qwen)
    ├── Apply LoRA/QLoRA
    └── Enable gradient checkpointing
    ↓
[Create Trainer]
    ├── SFTTrainer from TRL
    ├── BF16 mixed precision
    └── TensorBoard logging
    ↓
[Training Loop]
    ├── Forward pass
    ├── Backward pass
    ├── Gradient updates
    └── Validation
    ↓
[Save]
    ├── Checkpoint every N steps
    ├── Best model
    └── Training state
    ↓
outputs/checkpoints/
    ├── adapter_model/
    ├── trainer_state.json
    └── config.yaml
```

### Inference Flow

```
User Prompt
    ↓
[Format] System message + tools
    ↓
[Tokenize]
    ↓
[Generate]
    ├── Load base model + adapter
    ├── Forward pass
    └── Decode output
    ↓
Raw Output String
    ↓
[Parse JSON]
    ├── Extract JSON
    ├── Validate format
    └── Return tool call
    ↓
Tool Call JSON
    ↓
[Return Result]
    ├── Tool call data
    ├── Latency
    └── GPU memory
```

### Evaluation Flow

```
test.json (10k examples)
    ↓
[For each example]
    ├── Get ground truth tool call
    ├── Generate prediction
    ├── Parse prediction
    └── Compare
    ↓
Comparisons (10k)
    ↓
[Compute Metrics]
    ├── Function accuracy
    ├── Argument accuracy
    ├── Exact match
    └── JSON validity
    ↓
Evaluation Report
    ├── metrics.json
    └── predictions.json
```

## Configuration Inheritance

```
base.yaml (default)
    ↓
    ├─→ lora.yaml (extends base)
    │   └─→ Overrides for LoRA optimization
    │
    └─→ qlora.yaml (extends base)
        └─→ Overrides for QLoRA + 4-bit
```

## Extension Points

### 1. Add New Dataset

```python
class NewDatasetAdapter(DatasetAdapter):
    def download(self, cache_dir):
        # Implement
    def load_tools(self):
        # Implement
    # ... other methods
```

Then update `configs/base.yaml`:
```yaml
dataset:
  adapter: "new_dataset"
```

### 2. Add New Model

Update `configs/base.yaml`:
```yaml
model:
  name: "meta-llama/Llama-2-7b"
  torch_dtype: "bfloat16"
```

Everything else works unchanged.

### 3. Custom Training Strategy

Modify `train.py`:
```python
class ModelTrainer:
    def _setup_trainer(self, train_dataset, val_dataset):
        # Custom trainer setup
        pass
```

### 4. New Evaluation Metrics

Add to `evaluate.py`:
```python
def compute_custom_metric(predictions, ground_truth):
    # Custom metric computation
    pass
```

## Performance Characteristics

### Memory Usage (T4 - 16GB)

| Optimization | Batch | Gradient Accum | GPU Mem |
|--------------|-------|----------------|---------|
| LoRA         | 4     | 2              | ~12GB   |
| QLoRA        | 2     | 4              | ~8GB    |

### Training Time

- **LoRA**: ~6 hours for 3 epochs on 40k examples
- **QLoRA**: ~8 hours (slightly slower due to quantization)

### Inference Latency

- **Latency**: 0.25-0.35 seconds per example
- **Throughput**: ~150-200 examples/minute
- **GPU Memory**: 7-8GB during inference

## File Formats

### chat_format.json

```json
[
  {
    "messages": [
      {"role": "system", "content": "You are a helpful assistant..."},
      {"role": "user", "content": "Schedule a meeting..."},
      {"role": "assistant", "content": "{...tool call...}"}
    ],
    "tools": [...]
  },
  ...
]
```

### prediction.json

```json
{
  "ground_truth": {...},
  "predicted": {...},
  "latency": 0.32,
  "valid_json": true
}
```

### metrics.json

```json
{
  "total": 10000,
  "valid_json": 9950,
  "function_accuracy": 0.9425,
  "argument_accuracy": 0.9183,
  "exact_match": 0.8942
}
```

## Error Handling

### Graceful Degradation

```python
try:
    result = model.generate(...)
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        logger.error("GPU OOM. Use QLoRA or reduce batch size")
    else:
        logger.error(f"Generation failed: {e}")
    return {"error": str(e)}
```

### Validation

```python
def validate(self) -> bool:
    assert len(self.tool_schemas) > 0, "No tools loaded"
    assert len(self.examples["train"]) > 0, "No examples loaded"
    return True
```

## Testing Strategy

```
unit_tests/
├── test_adapters.py
├── test_config.py
├── test_preprocessing.py
└── test_inference.py

integration_tests/
├── test_full_pipeline.py
└── test_api.py
```

Run: `pytest tests/`

## Documentation

- `README.md` - Overview and quick start
- `WORKFLOW.md` - Complete workflow guide
- Code comments for complex logic
- Docstrings for all public APIs
- Type hints throughout

## Monitoring & Logging

### Log Levels

- **ERROR**: Training failures, OOM
- **WARNING**: Low memory warnings
- **INFO**: Progress, metrics, milestones
- **DEBUG**: Detailed debugging info

### Log Outputs

```
logs/
└── experiment_20240625_170000.log

outputs/tensorboard/
└── events.out.tfevents.*
```

## Security

- ✅ No hardcoded credentials
- ✅ `.env.example` for configuration
- ✅ `.gitignore` for sensitive files
- ✅ Input validation for all APIs
- ✅ No SQL/command injection vectors

---

**This architecture ensures the framework is professional, maintainable, and ready for production use.**
