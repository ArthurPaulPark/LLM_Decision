"""Configuration management with YAML support."""

import yaml
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    name: str
    adapter: str
    cache_dir: str
    train_split: str
    validation_split: str
    test_split: str


@dataclass
class ModelConfig:
    name: str
    model_type: str
    torch_dtype: str
    device_map: str
    trust_remote_code: bool


@dataclass
class TrainingConfig:
    output_dir: str
    num_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_steps: int
    gradient_checkpointing: bool
    mixed_precision: str
    save_strategy: str
    save_steps: int
    eval_strategy: str
    eval_steps: int
    save_total_limit: int
    seed: int
    completion_only_loss: bool = True   # assistant 응답 토큰만 loss 계산 (v2)


@dataclass
class OptimizationConfig:
    type: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list
    bias: str
    task_type: str
    load_in_4bit: bool = False
    bnb_4bit_use_double_quant: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"


@dataclass
class InferenceConfig:
    temperature: float
    max_new_tokens: int
    top_p: float
    top_k: int
    repetition_penalty: float
    do_sample: bool


@dataclass
class LoggingConfig:
    level: str
    log_dir: str
    tensorboard_dir: str
    wandb_project: str
    wandb_enabled: bool


@dataclass
class PathsConfig:
    data_raw: str
    data_processed: str
    models_dir: str
    outputs_dir: str
    scripts_dir: str


@dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    optimization: OptimizationConfig
    inference: InferenceConfig
    logging: LoggingConfig
    paths: PathsConfig
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file with inheritance support."""
        raw_config = cls._load_raw_yaml(config_path)
        
        return cls(
            dataset=DatasetConfig(**raw_config["dataset"]),
            model=ModelConfig(**raw_config["model"]),
            training=TrainingConfig(**raw_config["training"]),
            optimization=OptimizationConfig(**raw_config["optimization"]),
            inference=InferenceConfig(**raw_config["inference"]),
            logging=LoggingConfig(**raw_config["logging"]),
            paths=PathsConfig(**raw_config["paths"])
        )
        
    @classmethod
    def _load_raw_yaml(cls, config_path: str) -> dict:
        """Load raw YAML and recursively merge parent configs if 'extends' is set."""
        path = Path(config_path)
        with open(path, "r") as f:
            raw_config = yaml.safe_load(f)
            
        if "extends" in raw_config:
            parent_name = raw_config["extends"]
            parent_path = path.parent / parent_name
            parent_config = cls._load_raw_yaml(str(parent_path))
            raw_config = cls._deep_merge(parent_config, raw_config)
            
        return raw_config
        
    @staticmethod
    def _deep_merge(dict1: dict, dict2: dict) -> dict:
        """Recursively merge dict2 into dict1."""
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "dataset": asdict(self.dataset),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "optimization": asdict(self.optimization),
            "inference": asdict(self.inference),
            "logging": asdict(self.logging),
            "paths": asdict(self.paths)
        }
    
    def save(self, output_path: str) -> None:
        """Save configuration to YAML file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        logger.info(f"✓ Config saved to {output_path}")
    
    def create_directories(self) -> None:
        """Create all required directories."""
        dirs = [
            self.paths.data_raw,
            self.paths.data_processed,
            self.paths.models_dir,
            self.paths.outputs_dir,
            self.paths.scripts_dir,
            self.logging.log_dir,
            self.logging.tensorboard_dir,
            self.training.output_dir,
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info("✓ All directories created")
