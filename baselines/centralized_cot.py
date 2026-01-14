import torch
from torch.utils.data import DataLoader, ConcatDataset
from typing import Dict, Any

from models.fedcot_model import FedCoTModel
from training.local_trainer import LocalTrainer
from data.loaders import DatasetLoader
from utils.logging import log_metrics

def run_centralized_cot(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Baseline 1: Centralized CoT
    Train one model on concatenated full (non-partitioned) data.
    """
    # Load all datasets without partitioning
    loader = DatasetLoader(
        subsample_ratio=config['datasets'][0]['subsample_ratio'],
        model_name=config['model']['name']
    )
    all_datasets = loader.load_datasets()
    full_train = ConcatDataset([ds['train'] for ds in all_datasets])
    train_loader = DataLoader(full_train, batch_size=config['fl']['batch_size'], shuffle=True)

    model = FedCoTModel(config)
    # Disable alignment (no global prototype)
    trainer = LocalTrainer(
        config=config,
        model=model,
        dataloader=train_loader,
        global_prototype=torch.zeros(config['heads']['projector']['embed_dim'])  # Dummy, ignored
    )

    # Override to disable alignment loss
    trainer.losses.lambda2 = 0.0

    metrics = trainer.train(local_epochs=config['fl']['num_rounds'] * config['fl']['local_epochs'])
    log_metrics({"centralized_cot_final": metrics}, step=0)

    return metrics