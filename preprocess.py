#!/usr/bin/env python3
"""
Preprocessing pipeline for universal framework.

Responsibilities:
- Load dataset via adapter
- Analyze and print statistics
- Convert to chat format
- Support tool calling and function calling
- Save processed dataset for training
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any
import pickle
from tqdm import tqdm

from adapters import DatasetAdapter, get_adapter
from configs.config import Config
from configs.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


class Preprocessor:
    """Universal preprocessing pipeline."""
    
    def __init__(self, config: Config, max_samples: int = None):
        """Initialize preprocessor."""
        self.config = config
        self.max_samples = max_samples
        self.adapter: DatasetAdapter = self._load_adapter()
        self.logger = get_logger(self.__class__.__name__)
    
    def _load_adapter(self) -> DatasetAdapter:
        """Load appropriate dataset adapter dynamically."""
        adapter_name = self.config.dataset.adapter
        try:
            return get_adapter(adapter_name)
        except ValueError as e:
            raise ValueError(f"Failed to load adapter: {str(e)}")
    
    def run(self) -> None:
        """Execute full preprocessing pipeline."""
        self.logger.info("=" * 80)
        self.logger.info("PREPROCESSING PIPELINE")
        self.logger.info("=" * 80)
        
        # Step 1: Download dataset
        self.logger.info("\n[1/5] Downloading dataset...")
        self.adapter.download(self.config.dataset.cache_dir)
        
        # Step 2: Load tools
        self.logger.info("\n[2/5] Loading tool schemas...")
        tools = self.adapter.load_tools()
        self.logger.info(f"✓ Loaded {len(tools)} unique tools")
        
        # Step 3: Load and analyze examples
        self.logger.info("\n[3/5] Loading and analyzing examples...")
        self._load_and_analyze_examples()
        
        # Step 4: Convert to chat format
        self.logger.info("\n[4/5] Converting to chat format...")
        self._convert_and_save_splits(tools)
        
        # Step 5: Print statistics
        self.logger.info("\n[5/5] Dataset statistics:")
        self._print_statistics()
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("✓ PREPROCESSING COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 80)
    
    def _load_and_analyze_examples(self) -> None:
        """Load examples and print structure."""
        for split in ["train", "validation", "test"]:
            try:
                examples = self.adapter.load_examples(split)
                if examples:
                    self.logger.info(f"✓ Loaded {len(examples)} examples from {split}")
                    
                    # Show first example structure
                    first_example = examples[0]
                    self.logger.info(f"\nExample from {split}:")
                    self.logger.info(f"  Prompt: {first_example.prompt[:100]}...")
                    self.logger.info(f"  Tool Call: {json.dumps(first_example.tool_call, indent=2)}")
            except Exception as e:
                self.logger.warning(f"Could not load {split}: {str(e)}")
    
    def _convert_and_save_splits(self, tools: List) -> None:
        """Convert examples to chat format and save."""
        Path(self.config.paths.data_processed).mkdir(parents=True, exist_ok=True)
        
        for split in ["train", "validation", "test"]:
            try:
                examples = self.adapter.examples.get(split, [])
                if not examples:
                    continue
                
                if self.max_samples is not None and self.max_samples > 0:
                    examples = examples[:self.max_samples]
                    self.logger.info(f"Limiting {split} split to {self.max_samples} examples")
                
                converted = []
                for example in tqdm(examples, desc=f"Converting {split}"):
                    chat_data = self.adapter.convert_to_chat_format(example, None)
                    converted.append(chat_data)
                
                # Save as JSON
                output_path = Path(self.config.paths.data_processed) / f"{split}.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(converted, f, ensure_ascii=False, indent=2)
                
                self.logger.info(f"✓ Saved {split} ({len(converted)} examples) to {output_path}")
                
            except Exception as e:
                self.logger.warning(f"Could not save {split}: {str(e)}")
        
        # Save tools reference
        tools_path = Path(self.config.paths.data_processed) / "tools.json"
        tools_data = [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "required": t.required_params
            }
            for t in tools
        ]
        with open(tools_path, "w", encoding="utf-8") as f:
            json.dump(tools_data, f, ensure_ascii=False, indent=2)
        self.logger.info(f"✓ Saved {len(tools)} tools to {tools_path}")
    
    def _print_statistics(self) -> None:
        """Print comprehensive dataset statistics."""
        stats = self.adapter.get_statistics()
        split_info = self.adapter.get_split_info()
        
        self.logger.info("\nDataset Information:")
        self.logger.info(f"  Dataset: {self.config.dataset.name}")
        self.logger.info(f"  Total Examples: {stats.get('total_examples', 'N/A')}")
        
        self.logger.info("\nSplit Distribution:")
        for split, count in split_info.items():
            pct = (count / sum(split_info.values())) * 100 if split_info else 0
            self.logger.info(f"  {split:12s}: {count:6d} ({pct:5.1f}%)")
        
        self.logger.info(f"\nTool Information:")
        self.logger.info(f"  Total Unique Tools: {stats.get('total_tools', 'N/A')}")
        
        if 'tools' in stats and stats['tools']:
            self.logger.info(f"  Sample Tools:")
            for tool in stats['tools'][:5]:
                self.logger.info(f"    - {tool['name']}")
        
        self.logger.info(f"\nProcessed Data Location:")
        self.logger.info(f"  Directory: {self.config.paths.data_processed}")
        self.logger.info(f"  Files: train.json, validation.json, test.json, tools.json")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Preprocess dataset for training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Config file path"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./data/cache",
        help="Cache directory for downloading datasets"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process per split"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        log_dir="./logs",
        level="INFO"
    )
    
    # Load config
    config = Config.from_yaml(args.config)
    config.dataset.cache_dir = args.cache_dir
    config.create_directories()
    
    # Run preprocessing
    preprocessor = Preprocessor(config, args.max_samples)
    preprocessor.run()


if __name__ == "__main__":
    main()
