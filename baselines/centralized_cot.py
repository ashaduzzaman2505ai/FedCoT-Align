import torch
from torch.utils.data import DataLoader, ConcatDataset
from typing import Dict, Any
from models.fedcot_model import FedCoTModel
from training.local_trainer import LocalTrainer

def run_centralized_cot(config: Dict[str, Any], datasets: List[torch.utils.data.Dataset]):
    """
    Upper Bound: Train on the full combined dataset without FL overhead.
    """
    # 1. Combine all client partitions into one big pool
    full_train = ConcatDataset(datasets)
    train_loader = DataLoader(full_train, batch_size=config['fl']['batch_size'], shuffle=True)
    
    model = FedCoTModel(config)
    
    # 2. Setup Trainer with Alignment disabled (lambda_align = 0)
    # We pass a dummy prototype because the code requires one, but we set weight to 0.
    config['loss_weights']['lambda_align'] = 0.0 
    
    trainer = LocalTrainer(
        config=config,
        model=model,
        dataloader=train_loader,
        global_prototype=torch.zeros(config['heads']['projector']['embed_dim'])
    )
    
    # Total epochs = rounds * local_epochs to match FL compute budget
    total_epochs = config['fl']['num_rounds'] * config['fl']['local_epochs']
    
    print(f"Starting Centralized Training for {total_epochs} epochs...")
    metrics = trainer.train(epochs=total_epochs)
    return model, metrics