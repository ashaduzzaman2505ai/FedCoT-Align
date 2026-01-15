from flwr.server.strategy import FedAvg
import flwr as fl

def get_standard_fl_strategy(config):
    """
    Baseline 2: Plain FedAvg.
    The clients will still have the heads, but we set loss weights to 0 
    via the config passed to the client_fn.
    """
    return FedAvg(
        fraction_fit=config['fl']['fraction_fit'],
        min_fit_clients=config['fl']['clients_per_round'],
        min_available_clients=config['fl']['num_clients'],
    )