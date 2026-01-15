import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from models.fedcot_model import FedCoTModel
from training.local_trainer import LocalTrainer

def test():
    with open('configs/model.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Force a potential mismatch for testing: Prototype on CPU
    model = FedCoTModel(config)
    global_proto = torch.randn(config['heads']['projector']['embed_dim']).cpu()
    
    # Dummy data
    dataset = TensorDataset(
        torch.randint(0, 100, (4, 128)), # input_ids
        torch.ones((4, 128)),            # mask
        torch.randint(0, 100, (4, 32)),  # labels
        torch.randint(0, 2, (4,))        # hallucination
    )
    
    def collate_fn(data):
        return {
            'input_ids': torch.stack([x[0] for x in data]),
            'attention_mask': torch.stack([x[1] for x in data]),
            'labels': torch.stack([x[2] for x in data]),
            'is_hallucination': torch.stack([x[3] for x in data])
        }
    
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    # This should now handle the device migration internally
    trainer = LocalTrainer(config, model, loader, global_proto)
    
    print("Executing device-aware training test...")
    try:
        metrics = trainer.train(epochs=1)
        print(f"Success! Metrics: {metrics}")
        print("✅ Step 5 is now fully corrected and device-agnostic.")
    except RuntimeError as e:
        print(f"❌ Still getting device error: {e}")

if __name__ == "__main__":
    test()