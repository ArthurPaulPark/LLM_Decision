"""FastAPI backend for dashboard."""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from configs.config import Config
from adapters import XLAMAdapter

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM Decision Framework API",
    description="API for function calling fine-tuning framework",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global config
CONFIG = None
ADAPTER = None


class ProjectInfo(BaseModel):
    """Project information."""
    name: str = "LLM Decision Framework"
    version: str = "1.0.0"
    description: str = "Universal framework for function calling fine-tuning"


class DatasetInfo(BaseModel):
    """Dataset information."""
    name: str
    total_examples: int
    splits: Dict[str, int]
    tools_count: int
    sample_tools: List[Dict[str, str]]


class TrainingConfig(BaseModel):
    """Training configuration."""
    model: str
    optimization: str
    batch_size: int
    learning_rate: float
    epochs: int
    lora_rank: int


class InferenceRequest(BaseModel):
    """Inference request."""
    prompt: str
    tools: Optional[List[Dict[str, Any]]] = None


class InferenceResponse(BaseModel):
    """Inference response."""
    tool_call: Dict[str, Any]
    latency_seconds: float
    gpu_memory_gb: float
    num_tokens: int


@app.get("/api/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/project", response_model=ProjectInfo, tags=["project"])
async def get_project_info():
    """Get project information."""
    return ProjectInfo()


@app.get("/api/config", tags=["config"])
async def get_config():
    """Get current configuration."""
    if CONFIG is None:
        raise HTTPException(status_code=404, detail="Configuration not loaded")
    return CONFIG.to_dict()


@app.post("/api/config/load", tags=["config"])
async def load_config(config_path: str):
    """Load configuration from file."""
    global CONFIG
    try:
        CONFIG = Config.from_yaml(config_path)
        CONFIG.create_directories()
        return {"status": "success", "message": f"Loaded config from {config_path}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/dataset/info", response_model=DatasetInfo, tags=["dataset"])
async def get_dataset_info():
    """Get dataset information."""
    if ADAPTER is None:
        raise HTTPException(status_code=404, detail="Adapter not initialized")
    
    stats = ADAPTER.get_statistics()
    split_info = ADAPTER.get_split_info()
    
    return DatasetInfo(
        name=ADAPTER.dataset_name,
        total_examples=stats.get("total_examples", 0),
        splits=split_info,
        tools_count=stats.get("total_tools", 0),
        sample_tools=stats.get("tools", [])[:5]
    )


@app.post("/api/dataset/download", tags=["dataset"])
async def download_dataset(cache_dir: str = "./data/cache"):
    """Download dataset."""
    global ADAPTER
    try:
        if ADAPTER is None:
            ADAPTER = XLAMAdapter()
        
        ADAPTER.download(cache_dir)
        return {
            "status": "success",
            "message": "Dataset downloaded",
            "splits": ADAPTER.get_split_info()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/dataset/preview", tags=["dataset"])
async def preview_dataset(split: str = "train", limit: int = 5):
    """Preview dataset examples."""
    if ADAPTER is None:
        raise HTTPException(status_code=404, detail="Adapter not initialized")
    
    try:
        examples = ADAPTER.load_examples(split)
        preview_examples = []
        
        for i, example in enumerate(examples[:limit]):
            preview_examples.append({
                "index": i,
                "prompt": example.prompt[:200],
                "tool_call": example.tool_call,
                "metadata": example.metadata
            })
        
        return {
            "split": split,
            "total": len(examples),
            "preview": preview_examples
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/models/list", tags=["models"])
async def list_models():
    """List available models."""
    models_dir = Path("./models")
    models = []
    
    if models_dir.exists():
        for model_dir in models_dir.glob("*/"):
            models.append({
                "name": model_dir.name,
                "path": str(model_dir),
                "size_mb": sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) / 1024 / 1024
            })
    
    return {"models": models}


@app.get("/api/adapters/list", tags=["models"])
async def list_adapters():
    """List available LoRA adapters."""
    adapters_dir = Path("./outputs/adapters")
    adapters = []
    
    if adapters_dir.exists():
        for adapter_dir in adapters_dir.glob("*/"):
            adapters.append({
                "name": adapter_dir.name,
                "path": str(adapter_dir),
                "created": adapter_dir.stat().st_mtime
            })
    
    return {"adapters": adapters}


@app.get("/api/training/status", tags=["training"])
async def get_training_status():
    """Get training status."""
    checkpoint_dir = Path("./outputs/checkpoints")
    
    if not checkpoint_dir.exists():
        return {
            "status": "idle",
            "message": "No training in progress"
        }
    
    # Look for trainer_state.json
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    
    if trainer_state_path.exists():
        with open(trainer_state_path) as f:
            trainer_state = json.load(f)
        
        return {
            "status": "completed",
            "current_step": trainer_state.get("global_step", 0),
            "best_metric": trainer_state.get("best_metric")
        }
    
    return {"status": "idle"}


@app.get("/api/evaluation/latest", tags=["evaluation"])
async def get_latest_evaluation():
    """Get latest evaluation results."""
    eval_dir = Path("./outputs/evaluation")
    
    if not eval_dir.exists():
        return {"message": "No evaluation found"}
    
    metrics_file = eval_dir / "metrics.json"
    
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        return metrics
    
    return {"message": "No metrics found"}


@app.get("/api/logs/latest", tags=["logs"])
async def get_latest_logs(lines: int = 100):
    """Get latest logs."""
    log_dir = Path("./logs")
    
    if not log_dir.exists():
        return {"logs": []}
    
    # Find latest log file
    log_files = list(log_dir.glob("experiment_*.log"))
    
    if not log_files:
        return {"logs": []}
    
    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest_log) as f:
        all_lines = f.readlines()
    
    return {
        "file": latest_log.name,
        "logs": all_lines[-lines:]
    }


@app.get("/api/tensorboard/url", tags=["monitoring"])
async def get_tensorboard_url():
    """Get TensorBoard URL."""
    return {
        "url": "http://localhost:6006",
        "tensorboard_dir": "./outputs/tensorboard"
    }


@app.post("/api/inference/generate", response_model=InferenceResponse, tags=["inference"])
async def generate_inference(request: InferenceRequest):
    """Generate tool call."""
    raise HTTPException(
        status_code=501,
        detail="Inference requires loaded model. Use inference.py script instead."
    )


@app.get("/api/stats", tags=["stats"])
async def get_stats():
    """Get framework statistics."""
    stats = {
        "dataset": None,
        "models": 0,
        "adapters": 0,
        "training_checkpoints": 0,
        "evaluations": 0
    }
    
    # Count resources
    if Path("./models").exists():
        stats["models"] = len(list(Path("./models").glob("*/")))
    
    if Path("./outputs/adapters").exists():
        stats["adapters"] = len(list(Path("./outputs/adapters").glob("*/")))
    
    if Path("./outputs/checkpoints").exists():
        stats["training_checkpoints"] = len(list(Path("./outputs/checkpoints").glob("checkpoint-*")))
    
    if Path("./outputs/evaluation").exists():
        stats["evaluations"] = len(list(Path("./outputs/evaluation").glob("metrics.json")))
    
    return stats


if __name__ == "__main__":
    import uvicorn
    
    # Load default config
    CONFIG = Config.from_yaml("configs/base.yaml")
    ADAPTER = XLAMAdapter()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
