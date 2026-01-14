import flwr as fl
from fed.strategy import FedAvg  # Use plain FedAvg

def run_standard_fl(config: Dict[str, Any]):
    """
    Baseline 2: Standard FedAvg (no CoT alignment, no verifier).
    Reuse Flower but disable custom parts.
    """
    # Similar to main server but use vanilla FedAvg
    strategy = FedAvg(
        fraction_fit=1.0,
        min_fit_clients=config['fl']['clients_per_round'],
        min_available_clients=config['fl']['num_clients']
    )

    # Start server with vanilla strategy (clients still use model, but losses ignore alignment/verifier)
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=config['fl']['num_rounds']),
        strategy=strategy
    )
    # In client, set lambda1=lambda2=0.0 via config override