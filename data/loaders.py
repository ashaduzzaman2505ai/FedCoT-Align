from typing import Dict, List, Any
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import yaml
import os

from utils.logging import setup_logging  # From Step 1

class DatasetLoader:
    """
    Loader for datasets with tokenization and formatting for CoT.

    Args:
        config_path (str): Path to datasets.yaml.
        subsample_ratio (float): Ratio for subsampling (e.g., 0.1 for T4).
        model_name (str): Model for tokenizer (e.g., 'google/t5-small').
        max_length (int): Tokenization max length.
    """
    def __init__(self, config_path: str = 'configs/datasets.yaml', subsample_ratio: float = 1.0,
                 model_name: str = 'google/t5-small', max_length: int = 512):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.subsample_ratio = subsample_ratio
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.datasets: List[DatasetDict] = []

    def load_datasets(self) -> List[DatasetDict]:
        """
        Load and preprocess all datasets.

        Returns:
            List[DatasetDict]: Loaded datasets with splits.
        """
        for ds_config in self.config['datasets']:
            ds = load_dataset(ds_config['path'], split=ds_config.get('split'))
            if isinstance(ds, DatasetDict):
                pass  # Already split
            else:
                ds = ds.train_test_split(test_size=0.2)  # Default split if not provided
            # Subsample for testing
            for split in ds:
                num_samples = int(len(ds[split]) * self.subsample_ratio * ds_config.get('subsample_ratio', 1.0))
                ds[split] = ds[split].shuffle().select(range(num_samples))
            # Tokenize and format
            ds = ds.map(self._preprocess, batched=True)
            self.datasets.append(ds)
        return self.datasets

    def _preprocess(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        Preprocess function for tokenization and CoT prompting.

        Args:
            examples (Dict[str, List[Any]]): Batch of examples.

        Returns:
            Dict[str, List[Any]]: Tokenized inputs/labels.
        """
        # Dataset-specific formatting (simplified; extend as needed)
        if 'question' in examples:  # GSM8K, HotpotQA
            inputs = [f"Question: {q} Let's think step by step." for q in examples['question']]
            labels = examples.get('answer', examples.get('answers', ['']))  # Adjust keys
        elif 'query' in examples:  # TruthfulQA (assuming 'generation' split has 'query')
            inputs = [f"Query: {q} Reason factually." for q in examples['query']]
            labels = examples.get('best_answer', [''])
        else:
            raise ValueError("Unknown dataset format.")

        model_inputs = self.tokenizer(inputs, max_length=self.max_length, truncation=True, padding='max_length')
        with self.tokenizer.as_target_tokenizer():
            model_inputs['labels'] = self.tokenizer(labels, max_length=self.max_length, truncation=True, padding='max_length')['input_ids']
        
        # Placeholder for hallucination labels (binary; simulate for now, extend with NLI later)
        model_inputs['hallucination_labels'] = [0] * len(inputs)  # 0: no hallucination; generate properly in eval

        return model_inputs

# Example usage (for testing): loader = DatasetLoader(subsample_ratio=0.01); datasets = loader.load_datasets()