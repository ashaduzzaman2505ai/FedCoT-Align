import flwr as fl
from typing import Dict
from .strategy import FedCoTAlignStrategy

def start_server(config: Dict[str, Any], global_prototype_init: torch.Tensor):
    """
    Start Flower server with custom FedCoT-Align strategy.
    """
    strategy = FedCoTAlignStrategy(
        fraction_fit=1.0,  # Adjust via config
        fraction_evaluate=0.0,
        min_fit_clients=config['fl']['clients_per_round'],
        min_available_clients=config['fl']['num_clients'],
        global_prototype=global_prototype_init,
        embed_dim=config['heads']['projector']['embed_dim']
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=config['fl']['num_rounds']),
        strategy=strategy,
    )