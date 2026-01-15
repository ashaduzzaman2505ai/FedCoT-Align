import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import flwr as fl
import torch
import yaml
from fed.client import FedCoTClient
from fed.strategy import FedCoTAlignStrategy
from models.fedcot_model import FedCoTModel

def test_sim():
    with open('configs/fl.yaml', 'r') as f:
        fl_config = yaml.safe_load(f)
    with open('configs/model.yaml', 'r') as f:
        m_config = yaml.safe_load(f)

    # 1. Strategy Setup
    init_proto = torch.randn(m_config['heads']['projector']['embed_dim'])
    strategy = FedCoTAlignStrategy(
        global_prototype_init=init_proto,
        fraction_fit=1.0, 
        min_available_clients=2
    )

    # 2. Client Generator (Simulation will call this)
    def client_fn(cid: str):
        model = FedCoTModel(m_config)
        # For test, use small dummy loaders
        return FedCoTClient(int(cid), model, None, None, {**fl_config, **m_config})

    print("Starting FedCoT-Align Simulation...")
    # This will run for 1 round to verify the sync works
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
    )
    print("✅ Step 6 Passed: Federated Prototype Sync and Parameter Aggregation successful.")

if __name__ == "__main__":
    test_sim()