import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from models.fedcot_model import FedCoTModel
from training.local_trainer import LocalTrainer

def test():
    with open('configs/model.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup dummy model and prototype
    model = FedCoTModel(config)
    global_proto = torch.randn(config['heads']['projector']['embed_dim'])
    
    # Create dummy data: 8 samples
    dummy_input = torch.randint(0, 100, (8, 128))
    dummy_mask = torch.ones((8, 128))
    dummy_labels = torch.randint(0, 100, (8, 32))
    dummy_halluc = torch.randint(0, 2, (8,))
    
    dataset = TensorDataset(dummy_input, dummy_mask, dummy_labels, dummy_halluc)
    # Convert list of tensors to dict-style batches like HF Loader
    def collate_fn(data):
        return {
            'input_ids': torch.stack([x[0] for x in data]),
            'attention_mask': torch.stack([x[1] for x in data]),
            'labels': torch.stack([x[2] for x in data]),
            'is_hallucination': torch.stack([x[3] for x in data])
        }
    
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    # Initialize Trainer
    trainer = LocalTrainer(config, model, loader, global_proto)
    
    print("Starting single-epoch test train...")
    metrics = trainer.train(epochs=1)
    
    assert metrics['train_total'] > 0
    assert metrics['train_alignment'] > 0
    print(f"Metrics: {metrics}")
    print("✅ Step 5 Passed: Trainer and Tripartite Loss are fully operational.")

if __name__ == "__main__":
    test()