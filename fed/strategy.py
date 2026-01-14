import flwr as fl
from flwr.common import Parameters, FitRes, Scalar
from flwr.server.strategy import FedAvg
from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np

class FedCoTAlignStrategy(FedAvg):
    """
    Custom FedAvg extension for FedCoT-Align.
    Aggregates model weights (standard FedAvg) + CoT prototypes (mean aggregation).
    """
    def __init__(
        self,
        *args,
        global_prototype: torch.Tensor,
        embed_dim: int,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.global_prototype = global_prototype
        self.embed_dim = embed_dim

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if not results:
            return aggregated_parameters, aggregated_metrics

        # Aggregate local prototypes → new global prototype
        prototypes = []
        weights = []
        for _, fit_res in results:
            proto_np = fit_res.metrics.get("local_prototype")
            if proto_np is not None:
                prototypes.append(torch.tensor(proto_np))
                weights.append(fit_res.num_examples)

        if prototypes:
            weighted_prototypes = torch.stack(prototypes) * torch.tensor(weights).view(-1, 1)
            self.global_prototype = weighted_prototypes.sum(dim=0) / sum(weights)

        aggregated_metrics["global_prototype_norm"] = torch.norm(self.global_prototype).item()

        return aggregated_parameters, aggregated_metrics

    def get_global_prototype(self) -> torch.Tensor:
        return self.global_prototype