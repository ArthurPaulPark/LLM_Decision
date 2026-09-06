"""Registry pattern for dynamically loading dataset adapters."""

import logging
from typing import Dict, Type
from adapters.base_adapter import DatasetAdapter

logger = logging.getLogger(__name__)

_REGISTRY: Dict[str, Type[DatasetAdapter]] = {}


def register_adapter(name: str):
    """Decorator to register a dataset adapter."""
    def decorator(cls: Type[DatasetAdapter]):
        _REGISTRY[name.lower()] = cls
        logger.info(f"Registered adapter: {name} -> {cls.__name__}")
        return cls
    return decorator


def get_adapter(name: str, *args, **kwargs) -> DatasetAdapter:
    """Factory function to get an instance of a registered adapter."""
    cls = _REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown adapter: {name}. Available: {list(_REGISTRY.keys())}"
        )
    return cls(*args, **kwargs)
