import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import yaml
from models.fedcot_model import FedCoTModel

def test():
    # Load configs
    with open('configs/model.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize Model (T5-small is lightweight for testing)
    print("Initializing FedCoTModel...")
    model = FedCoTModel(config)
    
    # Create Dummy Batch
    batch_size = 4
    input_ids = torch.randint(0, 2000, (batch_size, 128))
    attention_mask = torch.ones((batch_size, 128))
    labels = torch.randint(0, 2000, (batch_size, 32))
    
    # Test Forward Pass
    print("Testing forward pass...")
    outputs = model(input_ids, attention_mask, labels)
    
    # Validation Checks
    assert 'answer_logits' in outputs, "Missing answer logits"
    assert outputs['cot_embedding'].shape == (batch_size, config['heads']['projector']['embed_dim']), "CoT embedding dim mismatch"
    assert outputs['verifier_logits'].shape == (batch_size, config['heads']['verifier']['num_classes']), "Verifier logits dim mismatch"
    
    print("✅ Step 4 Passed: Model architecture is modular and dimensions are correct.")

if __name__ == "__main__":
    test()