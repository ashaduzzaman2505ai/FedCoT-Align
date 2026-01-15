import flwr as fl
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from fed.client import FedCoTClient
from fed.strategy import FedCoTAlignStrategy
from models.fedcot_model import FedCoTModel

def test_sim():
    with open('configs/fl.yaml', 'r') as f: fl_cfg = yaml.safe_load(f)
    with open('configs/model.yaml', 'r') as f: m_cfg = yaml.safe_load(f)

    # 1. Create a tiny dummy dataset for the simulation to use
    def get_dummy_loader():
        dataset = TensorDataset(
            torch.randint(0, 100, (10, 128)), # input_ids
            torch.ones((10, 128)),            # attention_mask
            torch.randint(0, 100, (10, 32)),  # labels
            torch.randint(0, 2, (10,))        # is_hallucination
        )
        def collate_fn(data):
            return {
                'input_ids': torch.stack([x[0] for x in data]),
                'attention_mask': torch.stack([x[1] for x in data]),
                'labels': torch.stack([x[2] for x in data]),
                'is_hallucination': torch.stack([x[3] for x in data])
            }
        return DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    # 2. Strategy Setup
    init_proto = torch.zeros(m_cfg['heads']['projector']['embed_dim'])
    strategy = FedCoTAlignStrategy(
        global_prototype_init=init_proto,
        fraction_fit=1.0, 
        min_available_clients=2
    )

    # 3. Client Generator
    def client_fn(cid: str):
        # IMPORTANT: Do not pass None for trainloader
        train_loader = get_dummy_loader()
        model = FedCoTModel(m_cfg)
        
        # Ensure config has 'fl' key to avoid the KeyError we saw earlier
        combined_config = {**m_cfg, 'fl': fl_cfg['fl']}
        
        return FedCoTClient(
            int(cid), model, train_loader, None, combined_config
        ).to_client()

    print("Starting FedCoT-Align Simulation...")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
        ray_init_args={"num_cpus": 2, "ignore_reinit_error": True}
    )

if __name__ == "__main__":
    test_sim()