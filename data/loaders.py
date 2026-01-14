from typing import Dict, List, Any
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import yaml
import torch

class DatasetLoader:
    def __init__(self, config_path: str, model_name: str, max_length: int = 512, subsample_size: int = -1):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.subsample_size = subsample_size

    def _get_keys(self, ds_name: str):
        """Maps dataset names to their specific Question/Answer keys."""
        mapping = {
            "gsm8k": ("question", "answer"),
            "truthfulqa": ("question", "best_answer"),
            "hotpotqa": ("question", "answer")
        }
        return mapping.get(ds_name.lower(), ("question", "answer"))

    def load_and_preprocess(self) -> List[Dataset]:
        processed_datasets = []
        for ds_cfg in self.config['data']['sources']:
            print(f"Loading {ds_cfg['name']}...")
            ds = load_dataset(ds_cfg['path'], ds_cfg.get('config'), split='train')
            
            if self.subsample_size > 0:
                n = min(self.subsample_size, len(ds))
                ds = ds.shuffle(seed=42).select(range(n))

            q_key, a_key = self._get_keys(ds_cfg['name'])
            
            def tokenize_fn(examples):
                # Apply CoT Prompt Template
                inputs = [f"Question: {q} Let's think step by step." for q in examples[q_key]]
                # Ensure targets are strings (TruthfulQA can sometimes be list)
                targets = [str(a) for a in examples[a_key]]

                model_inputs = self.tokenizer(
                    inputs, max_length=self.max_length, truncation=True, padding='max_length'
                )
                labels = self.tokenizer(
                    text_target=targets, max_length=128, truncation=True, padding='max_length'
                )
                
                # Replace padding token id with -100 so it's ignored by the loss function
                model_inputs["labels"] = [
                    [(l if l != self.tokenizer.pad_token_id else -100) for l in label_seq]
                    for label_seq in labels["input_ids"]
                ]
                
                # Synthetic hallucination labels for Verifier training
                # Logic: In real scenarios, this comes from ground-truth or NLI. 
                # Here we initialize as 0 (correct).
                model_inputs["is_hallucination"] = [0] * len(inputs)
                return model_inputs

            ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
            ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "is_hallucination"])
            processed_datasets.append(ds)
            
        return processed_datasets