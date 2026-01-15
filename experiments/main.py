import argparse
import yaml
import os
import torch
import flwr as fl
from typing import Dict, Any
from pathlib import Path

from data.loaders import DatasetLoader
from data.partitioner import NonIIDPartitioner
from models.fedcot_model import FedCoTModel
from fed.client import FedCoTClient
from fed.strategy import FedCoTAlignStrategy
from utils.seeding import set_seed
from utils.logging import setup_logging

def run_fedcot_align(config: Dict[str, Any]):
    set_seed(config['fl']['seed'])
    
    # 1. Setup Logging (W&B or Console)
    setup_logging(
        use_wandb=config.get('experiment', {}).get('use_wandb', False), 
        config=config,
        project_name="FedCoT-Align"
    )

    # 2. Data Preparation
    print("--- Loading and Partitioning Data ---")
    loader = DatasetLoader('configs/datasets.yaml', config['model']['name'])
    raw_datasets = loader.load_and_preprocess()
    
    partitioner = NonIIDPartitioner(
        raw_datasets, 
        num_clients=config['fl']['num_clients'], 
        alpha=config['partitioning'].get('skew_level', 0.1)
    )
    client_datasets = partitioner.partition()

    # 3. Strategy Initialization (The Brain of the Server)
    init_proto = torch.zeros(config['heads']['projector']['embed_dim'])
    strategy = FedCoTAlignStrategy(
        global_prototype_init=init_proto,
        fraction_fit=config['fl']['fraction_fit'],
        min_fit_clients=config['fl']['clients_per_round'],
        min_available_clients=config['fl']['num_clients'],
        # Pass hyperparams to strategy for logging
        initial_parameters=None 
    )

    # 4. Client Factory (Virtualizes clients for Ray)
    def client_fn(cid: str):
        cid_int = int(cid)
        # Ensure each client gets its subset
        train_loader = torch.utils.data.DataLoader(
            client_datasets[cid_int], 
            batch_size=config['fl']['batch_size'], 
            shuffle=True
        )
        
        # Load local model (LoRA-wrapped)
        model = FedCoTModel(config)
        
        # We ensure the config has all sub-sections for the client
        return FedCoTClient(
            cid_int, model, train_loader, None, config
        ).to_client()

    # 5. Launch Flower Simulation (Ray Backend)
    print(f"--- Launching Simulation: {config['fl']['num_rounds']} rounds ---")
    
    # T4 GPU Allocation logic: 0.5 means 2 clients can share one GPU
    gpu_resource = 0.5 if torch.cuda.is_available() else 0
    
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config['fl']['num_clients'],
        config=fl.server.ServerConfig(num_rounds=config['fl']['num_rounds']),
        strategy=strategy,
        ray_init_args={
            "num_cpus": 2, 
            "num_gpus": gpu_resource,
            "ignore_reinit_error": True
        }
    )

def main():
    parser = argparse.ArgumentParser(description="FedCoT-Align Entry Point")
    parser.add_argument('--config', type=str, nargs='+', 
                        default=['configs/model.yaml', 'configs/fl.yaml', 'configs/datasets.yaml'])
    parser.add_argument('--baselines', action='store_true')
    args = parser.parse_args()

    # Deep Merge Logic
    config = {}
    for c_path in args.config:
        with open(c_path, 'r') as f:
            config.update(yaml.safe_load(f))

    if args.baselines:
        print("Running Baseline: Centralized CoT")
        # run_centralized_cot(config)
    else:
        run_fedcot_align(config)

if __name__ == "__main__":
    main()