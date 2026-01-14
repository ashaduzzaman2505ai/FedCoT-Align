import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.seeding import set_seed
from utils.logging import setup_logging, log_metrics
from utils.checkpointing import save_checkpoint
import torch
import torch.nn as nn
import yaml

def test():
    # Test Seeding
    set_seed(42)
    a = torch.randn(2)
    set_seed(42)
    b = torch.randn(2)
    assert torch.equal(a, b), "Deterministic seeding failed!"

    # Test Logger
    logger = setup_logging(use_wandb=False)
    logger.info("Local setup verification successful.")

    # Test Config Loading
    with open("configs/model.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        assert cfg['model']['name'] == "google/flan-t5-small"
    
    print("✅ Step 1 & 2 passed: Environment is reproducible and configs are valid.")

if __name__ == "__main__":
    test()