import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any
import logging

from models.fedcot_model import FedCoTModel
from training.losses import FedCoTLosses
from utils.logging import log_metrics
from utils.checkpointing import save_checkpoint

class LocalTrainer:
    def __init__(self, config: Dict[str, Any], model: FedCoTModel, dataloader: DataLoader, 
                 global_prototype: torch.Tensor):
        self.config = config
        self.model = model
        self.dataloader = dataloader
        
        # --- DEVICE FIX ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Ensure the global prototype is on the SAME device as the model
        self.global_prototype = global_prototype.to(self.device)
        self.model.to(self.device)
        # ------------------
        
        self.losses = FedCoTLosses(
            lambda_verifier=config['loss_weights']['lambda_verifier'],
            lambda_align=config['loss_weights']['lambda_align']
        )
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=float(config['optimizer']['lr']),
            weight_decay=float(config['optimizer']['weight_decay'])
        )
        
        self.logger = logging.getLogger("fedcot")

    def train(self, epochs: int) -> Dict[str, float]:
        self.model.train()
        final_metrics = {}

        for epoch in range(epochs):
            running_metrics = {'total': 0.0, 'answer': 0.0, 'verifier': 0.0, 'alignment': 0.0}
            
            for batch in self.dataloader:
                # Move all batch tensors to the correct device (GPU)
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()
                }
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                # Compute Tripartite Loss 
                # (Inside compute, all tensors: outputs, batch, and global_prototype are now on GPU)
                loss_dict = self.losses.compute(outputs, batch, self.global_prototype)
                
                # Backward pass
                loss_dict['total'].backward()
                self.optimizer.step()
                
                for k in running_metrics:
                    running_metrics[k] += loss_dict[k].item()

            n = len(self.dataloader)
            final_metrics = {f"train_{k}": v / n for k, v in running_metrics.items()}
            log_metrics(final_metrics, step=epoch)

        return final_metrics