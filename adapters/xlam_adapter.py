"""xLAM dataset adapter for Salesforce xlam-function-calling-60k."""

import json
import logging
from typing import Any, Dict, List, Optional
from datasets import load_dataset
from adapters.base_adapter import DatasetAdapter, ToolSchema, Example
from adapters.registry import register_adapter

logger = logging.getLogger(__name__)


@register_adapter("xlam")
class XLAMAdapter(DatasetAdapter):
    """Adapter for Salesforce xlam-function-calling-60k dataset."""
    
    def __init__(self):
        super().__init__(
            dataset_name="Salesforce/xlam-function-calling-60k",
            version="1.0"
        )
        self.dataset = None
    
    def download(self, cache_dir: str) -> None:
        """Download xLAM dataset from HuggingFace.

        xlam-function-calling-60k는 'train' 스플릿만 존재하므로
        자동으로 train(90%) / validation(5%) / test(5%)로 분할합니다.
        """
        logger.info(f"Downloading {self.dataset_name}...")
        raw = load_dataset(
            self.dataset_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        logger.info(f"✓ Downloaded successfully. Original splits: {list(raw.keys())}")

        # xlam 데이터셋은 'train' 스플릿만 제공 → 자동 분할
        if "validation" not in raw and "test" not in raw:
            logger.info("Only 'train' split found. Auto-splitting: 90/5/5...")
            split_result = raw["train"].train_test_split(test_size=0.10, seed=42)
            val_test = split_result["test"].train_test_split(test_size=0.50, seed=42)
            from datasets import DatasetDict
            self.dataset = DatasetDict({
                "train":      split_result["train"],
                "validation": val_test["train"],
                "test":       val_test["test"],
            })
            logger.info(
                f"✓ Split result → train: {len(self.dataset['train'])}, "
                f"validation: {len(self.dataset['validation'])}, "
                f"test: {len(self.dataset['test'])}"
            )
        else:
            self.dataset = raw
    
    def load_tools(self) -> List[ToolSchema]:
        """Extract tool schemas from xLAM dataset."""
        if self.dataset is None:
            raise RuntimeError("Dataset not downloaded. Call download() first.")
        
        self.tool_schemas = []
        seen_tools = set()
        
        for split in ["train", "validation", "test"]:
            if split not in self.dataset:
                continue
                
            for example in self.dataset[split]:
                if "tools" in example:
                    tools = example["tools"]
                    if isinstance(tools, str):
                        tools = json.loads(tools)
                    
                    if isinstance(tools, list):
                        for tool in tools:
                            tool_name = tool.get("name", "")
                            if tool_name and tool_name not in seen_tools:
                                seen_tools.add(tool_name)
                                self.tool_schemas.append(
                                    ToolSchema(
                                        name=tool_name,
                                        description=tool.get("description", ""),
                                        parameters=tool.get("parameters", {}),
                                        required_params=tool.get("required", [])
                                    )
                                )
        
        logger.info(f"✓ Loaded {len(self.tool_schemas)} unique tools")
        return self.tool_schemas
    
    def load_examples(self, split: str = "train") -> List[Example]:
        """Load training examples from xLAM dataset."""
        if self.dataset is None:
            raise RuntimeError("Dataset not downloaded. Call download() first.")
        
        if split not in self.dataset:
            raise ValueError(f"Split '{split}' not found. Available: {list(self.dataset.keys())}")
        
        examples = []
        for item in self.dataset[split]:
            example = Example(
                prompt=item.get("query", ""),
                tool_call=self._parse_tool_call(item),
                metadata={
                    "id": item.get("id", ""),
                    "dataset": "xlam",
                    "tools": item.get("tools", [])
                }
            )
            examples.append(example)
        
        self.examples[split] = examples
        logger.info(f"✓ Loaded {len(examples)} examples from {split} split")
        return examples
    
    def convert_to_chat_format(
        self, 
        example: Example, 
        tools: Optional[List[ToolSchema]] = None
    ) -> Dict[str, Any]:
        """Convert xLAM example to Qwen chat format."""
        if tools is None:
            raw_tools = example.metadata.get("tools", [])
            if isinstance(raw_tools, str):
                try:
                    raw_tools = json.loads(raw_tools)
                except json.JSONDecodeError:
                    raw_tools = []
            
            parsed_tools = []
            if isinstance(raw_tools, list):
                for t in raw_tools:
                    if isinstance(t, dict):
                        parsed_tools.append(
                            ToolSchema(
                                name=t.get("name", ""),
                                description=t.get("description", ""),
                                parameters=t.get("parameters", {}),
                                required_params=t.get("required", [])
                            )
                        )
            elif isinstance(raw_tools, dict):
                parsed_tools.append(
                    ToolSchema(
                        name=raw_tools.get("name", ""),
                        description=raw_tools.get("description", ""),
                        parameters=raw_tools.get("parameters", {}),
                        required_params=raw_tools.get("required", [])
                    )
                )
            tools = parsed_tools

        tool_descriptions = ""
        if tools:
            tool_descriptions = json.dumps(
                [
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters
                    }
                    for t in tools
                ],
                indent=2,
                ensure_ascii=False
            )
        
        messages = [
            {
                "role": "system",
                "content": f"You are a helpful assistant with access to the following tools:\n{tool_descriptions}"
            },
            {
                "role": "user",
                "content": example.prompt
            },
            {
                "role": "assistant",
                "content": json.dumps(example.tool_call, ensure_ascii=False)
            }
        ]
        
        formatted_tools = []
        if tools:
            for t in tools:
                formatted_tools.append({
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                    "required": t.required_params
                })
        
        return {
            "messages": messages,
            "tools": formatted_tools
        }
    
    def convert_to_inference_format(
        self, 
        prompt: str, 
        tools: Optional[List[ToolSchema]] = None
    ) -> List[Dict[str, str]]:
        """Convert prompt and tools to inference chat messages format (system, user)."""
        tool_descriptions = ""
        if tools:
            # Check if tools are already ToolSchema objects or dicts
            formatted_tools = []
            for t in tools:
                if isinstance(t, ToolSchema):
                    formatted_tools.append({
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters
                    })
                elif isinstance(t, dict):
                    formatted_tools.append({
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {})
                    })
            tool_descriptions = json.dumps(
                formatted_tools,
                indent=2,
                ensure_ascii=False
            )
        
        return [
            {
                "role": "system",
                "content": f"You are a helpful assistant with access to the following tools:\n{tool_descriptions}"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    
    def parse_prediction(
        self, 
        output: str, 
        tools: Optional[List[ToolSchema]] = None
    ) -> Dict[str, Any]:
        """Parse model output to tool call."""
        try:
            tool_call = json.loads(output)
            
            # Validate tool name exists
            if "tool_name" in tool_call or "function" in tool_call:
                return tool_call
            
            return {"error": "Invalid format: missing tool_name or function"}
        except json.JSONDecodeError as e:
            return {"error": f"JSON decode error: {str(e)}", "output": output}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if self.dataset is None:
            return {"error": "Dataset not loaded"}
        
        stats = {
            "total_examples": sum(len(self.dataset[s]) for s in self.dataset.keys() if s in self.examples),
            "splits": {}
        }
        
        for split in ["train", "validation", "test"]:
            if split in self.dataset:
                stats["splits"][split] = len(self.dataset[split])
        
        if self.tool_schemas:
            stats["total_tools"] = len(self.tool_schemas)
            stats["tools"] = [
                {"name": t.name, "description": t.description}
                for t in self.tool_schemas[:10]  # First 10
            ]
        
        return stats
    
    def get_split_info(self) -> Dict[str, int]:
        """Get split information."""
        if self.dataset is None:
            return {}
        
        return {split: len(self.dataset[split]) for split in self.dataset.keys()}
    
    @staticmethod
    def _parse_tool_call(item: Dict[str, Any]) -> Dict[str, Any]:
        """Parse tool call from xLAM item."""
        answers = item.get("answers", "")
        if isinstance(answers, str):
            try:
                answers = json.loads(answers)
            except json.JSONDecodeError:
                return {"tool_name": "", "arguments": {}}
        
        if isinstance(answers, list) and len(answers) > 0:
            # Extract from the first tool call to match our single tool call framework
            first_call = answers[0]
            if isinstance(first_call, dict):
                return {
                    "tool_name": first_call.get("name", ""),
                    "arguments": first_call.get("arguments", {})
                }
        elif isinstance(answers, dict):
            return {
                "tool_name": answers.get("name", ""),
                "arguments": answers.get("arguments", {})
            }
        
        return {"tool_name": "", "arguments": {}}
