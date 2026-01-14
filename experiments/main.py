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
from fed.server import start_server
from fed.strategy import FedCoTAlignStrategy
from utils.seed import set_seed
from utils.logging import setup_logging, log_metrics
from baselines.centralized_cot import run_centralized_cot
# Import other baselines as needed

def load_and_merge_config(config_paths: list[str]) -> Dict[str, Any]:
    """Load and merge multiple YAML configs (later ones override)."""
    config = {}
    for path in config_paths:
        with open(path, 'r') as f:
            override = yaml.safe_load(f)
            # Simple recursive merge (extend for deep merge if needed)
            config.update(override)
    return config

def run_fedcot_align(config: Dict[str, Any]):
    """Run main FedCoT-Align experiment."""
    set_seed(config['fl']['seed'])
    setup_logging(use_wandb=config.get('experiment', {}).get('use_wandb', False), config=config)

    # Load & partition data
    loader = DatasetLoader(
        subsample_ratio=config['datasets'][0].get('subsample_ratio', 1.0),
        model_name=config['model']['name']
    )
    datasets = loader.load_datasets()
    partitioner = NonIIDPartitioner(
        datasets=datasets,
        num_clients=config['fl']['num_clients'],
        skew_level=config['partitioning']['skew_level'],
        seed=config['fl']['seed']
    )
    client_datasets = partitioner.partition()

    # Initialize global prototype (random or zero)
    global_prototype = torch.zeros(config['heads']['projector']['embed_dim'])

    # Define client_fn for Flower
    def client_fn(cid: str):
        cid_int = int(cid)
        ds = client_datasets[cid_int]
        train_loader = torch.utils.data.DataLoader(ds['train'], batch_size=config['fl']['batch_size'], shuffle=True)
        val_loader = torch.utils.data.DataLoader(ds.get('test', ds['train']), batch_size=config['fl']['batch_size'])

        model = FedCoTModel(config)
        return FedCoTClient(
            client_id=cid_int,
            model=model,
            trainloader=train_loader,
            valloader=val_loader,
            config=config,
            global_prototype=global_prototype
        )

    # Start Flower server with custom strategy
    strategy = FedCoTAlignStrategy(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=config['fl']['clients_per_round'],
        min_available_clients=config['fl']['num_clients'],
        global_prototype=global_prototype,
        embed_dim=config['heads']['projector']['embed_dim']
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=config['fl']['num_rounds']),
        strategy=strategy,
        client_fn=client_fn  # In simulation mode
    )

    log_metrics({"experiment": "fedcot_align_complete"}, step=0)

def main():
    parser = argparse.ArgumentParser(description="FedCoT-Align Experiment Runner")
    parser.add_argument('--config', type=str, nargs='+', default=['configs/model.yaml', 'configs/fl.yaml', 'configs/datasets.yaml'],
                        help='Paths to config YAML files (later override earlier)')
    parser.add_argument('--experiment', type=str, default='main_results',
                        help='Experiment name (e.g., main_results, table1, non_iid_ablation)')
    parser.add_argument('--baselines', action='store_true', help='Run baselines instead')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    config = load_and_merge_config(args.config)
    # Override with experiment-specific if exists
    exp_path = f"configs/experiments/{args.experiment}.yaml"
    if os.path.exists(exp_path):
        config = load_and_merge_config([*args.config, exp_path])

    if args.baselines:
        # Run selected baselines (extend as needed)
        run_centralized_cot(config)
        # Add calls to other baselines...
    else:
        run_fedcot_align(config)

if __name__ == "__main__":
    main()