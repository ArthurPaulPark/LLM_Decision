"""LLM Decision Framework - Universal function calling fine-tuning framework."""

__version__ = "1.0.0"
__author__ = "LLM Decision Framework Contributors"
__description__ = "Production-quality, reusable framework for function calling fine-tuning"

from configs.config import Config
from adapters import DatasetAdapter, XLAMAdapter

__all__ = [
    "Config",
    "DatasetAdapter",
    "XLAMAdapter",
]
