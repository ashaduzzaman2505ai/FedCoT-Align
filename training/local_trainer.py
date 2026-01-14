import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any
import yaml
from opacus import PrivacyEngine  # For DP-SGD; install if needed, but flag-gated

from models.fedcot_model import FedCoTModel
from training.losses import FedCoTLosses
from utils.logging import setup_logging, log_metrics
from utils.checkpointing import save_checkpoint, load_checkpoint
from utils.seeding import set_seed

class LocalTrainer:
    """
    Local training loop for each FL client.

    Args:
        config (Dict): Merged config from YAMLs.
        model (FedCoTModel): Client model.
        dataloader (DataLoader): Client data loader.
        global_prototype (torch.Tensor): Global CoT prototype.
    """
    def __init__(self, config: Dict[str, Any], model: FedCoTModel, dataloader: DataLoader,
                 global_prototype: torch.Tensor):
        self.config = config
        set_seed(config['fl']['seed'])
        self.model = model
        self.dataloader = dataloader
        self.losses = FedCoTLosses(
            lambda1=config['loss_weights']['lambda1'],
            lambda2=config['loss_weights']['lambda2']
        )
        self.optimizer = self._get_optimizer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.global_prototype = global_prototype.to(self.device)
        self.privacy_engine = None
        if config['privacy']['dp_sgd']:
            self._setup_dp()

        setup_logging(use_wandb=config.get('experiment', {}).get('use_wandb', False), config=config)

    def _get_optimizer(self) -> optim.Optimizer:
        opt_type = self.config['optimizer']['type']
        lr = self.config['optimizer']['lr']
        wd = self.config['optimizer']['weight_decay']
        if opt_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_type}")

    def _setup_dp(self):
        """Setup DP-SGD if flagged."""
        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.dataloader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dataloader,
            noise_multiplier=self.config['privacy']['noise_multiplier'],
            max_grad_norm=self.config['privacy']['max_grad_norm']
        )

    def train(self, local_epochs: int) -> Dict[str, float]:
        """
        Run local training for epochs.

        Args:
            local_epochs (int): Number of local epochs.

        Returns:
            Dict[str, float]: Average metrics.
        """
        self.model.train()
        metrics = {'loss': 0.0, 'answer_loss': 0.0, 'verifier_loss': 0.0, 'alignment_loss': 0.0}
        num_batches = 0

        for epoch in range(local_epochs):
            for batch in self.dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    hallucination_labels=batch['hallucination_labels']
                )
                loss_dict = self.losses.compute(
                    outputs, batch['labels'], batch['hallucination_labels'], self.global_prototype
                )
                loss = loss_dict['total']

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                for k in metrics:
                    metrics[k] += loss_dict.get(k, 0.0).item()
                num_batches += 1

            avg_metrics = {k: v / num_batches for k, v in metrics.items()}
            log_metrics(avg_metrics, epoch, use_wandb=self.config.get('experiment', {}).get('use_wandb', False))

        # Checkpoint
        save_checkpoint({'model_state': self.model.state_dict(), 'optimizer_state': self.optimizer.state_dict()},
                        path='checkpoints/', filename=f"client_{id(self)}.pth")

        return avg_metrics

    def load_from_checkpoint(self, path: str):
        """Load checkpoint if exists."""
        state = load_checkpoint(path)
        if state:
            self.model.load_state_dict(state['model_state'])
            self.optimizer.load_state_dict(state['optimizer_state'])