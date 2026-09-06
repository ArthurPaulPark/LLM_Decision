"""Base dataset adapter interface for universal compatibility."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolSchema:
    """Tool/Function schema representation."""
    name: str
    description: str
    parameters: Dict[str, Any]
    required_params: List[str]


@dataclass
class Example:
    """Single training example."""
    prompt: str
    tool_call: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class DatasetAdapter(ABC):
    """
    Abstract base class for dataset adapters.
    
    Only this class needs to be subclassed for new competition datasets.
    All core framework modules remain unchanged.
    """
    
    def __init__(self, dataset_name: str, version: str = "1.0"):
        """
        Initialize adapter.
        
        Args:
            dataset_name: Name of the dataset
            version: Dataset version
        """
        self.dataset_name = dataset_name
        self.version = version
        self.tool_schemas: List[ToolSchema] = []
        self.examples: Dict[str, List[Example]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        logger.info(f"Initialized {self.__class__.__name__} for {dataset_name} v{version}")
    
    @abstractmethod
    def download(self, cache_dir: str) -> None:
        """
        Download raw dataset.
        
        Args:
            cache_dir: Directory to cache the dataset
        """
        pass
    
    @abstractmethod
    def load_tools(self) -> List[ToolSchema]:
        """
        Load and parse tool/function schemas.
        
        Returns:
            List of tool schemas
        """
        pass
    
    @abstractmethod
    def load_examples(self, split: str = "train") -> List[Example]:
        """
        Load training examples for a specific split.
        
        Args:
            split: "train", "validation", or "test"
            
        Returns:
            List of examples
        """
        pass
    
    @abstractmethod
    def convert_to_chat_format(self, example: Example, tools: List[ToolSchema]) -> Dict[str, Any]:
        """
        Convert example to chat template format (e.g., Qwen format).
        
        Args:
            example: Single example
            tools: Available tools
            
        Returns:
            Chat format dictionary with "messages" and "tools"
        """
        pass
    
    @abstractmethod
    def convert_to_inference_format(
        self, 
        prompt: str, 
        tools: Optional[List[ToolSchema]] = None
    ) -> List[Dict[str, str]]:
        """
        Convert prompt and tools to inference chat messages format (system, user).
        
        Args:
            prompt: User prompt
            tools: Available tools
            
        Returns:
            List of chat messages containing system and user messages
        """
        pass

    
    @abstractmethod
    def parse_prediction(self, output: str, tools: List[ToolSchema]) -> Dict[str, Any]:
        """
        Parse model output into tool call.
        
        Args:
            output: Raw model output
            tools: Available tools for validation
            
        Returns:
            Parsed tool call dict or error dict
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics for analysis.
        
        Returns:
            Statistics dictionary
        """
        pass
    
    @abstractmethod
    def get_split_info(self) -> Dict[str, int]:
        """
        Get information about data splits.
        
        Returns:
            Dict with split names and example counts
        """
        pass
    
    def validate(self) -> bool:
        """
        Validate adapter configuration.
        
        Returns:
            True if valid
        """
        assert len(self.tool_schemas) > 0, "No tool schemas loaded"
        assert len(self.examples["train"]) > 0, "No training examples loaded"
        logger.info(f"✓ Adapter validation passed")
        return True
