import io
from flwr.server.strategy import FedAvg
from flwr.common import Scalar, FitRes, Parameters, parameters_to_ndarrays, ndarrays_to_parameters
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional


class FedCoTAlignStrategy(FedAvg):
    def __init__(self, global_prototype_init: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.global_prototype = global_prototype_init

    def configure_fit(self, server_round, parameters, client_manager):
        # Convert prototype to bytes for Flower compatibility
        proto_np = self.global_prototype.detach().cpu().numpy()
        proto_bytes = proto_np.tobytes() 
        
        config = {
            "global_prototype_bytes": proto_bytes, # Send as bytes
            "proto_shape": list(proto_np.shape),   # Shapes are usually small lists, but let's use a string or tuple if needed
            "server_round": server_round
        }
        
        instructions = super().configure_fit(server_round, parameters, client_manager)
        for _, fit_ins in instructions:
            fit_ins.config.update(config)
        return instructions

    def aggregate_fit(self, server_round, results, failures):
        # Standard Weight Aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if not results:
            return aggregated_parameters, aggregated_metrics

        # Prototype Aggregation (Weighted by number of local samples)
        protos, weights = [], []
        for _, fit_res in results:
            if "local_prototype" in fit_res.metrics:
                protos.append(np.array(fit_res.metrics["local_prototype"]))
                weights.append(fit_res.num_examples)
        
        if protos:
            protos_np = np.array(protos)
            weights_np = np.array(weights).reshape(-1, 1)
            new_proto_np = np.sum(protos_np * weights_np, axis=0) / np.sum(weights_np)
            self.global_prototype = torch.from_numpy(new_proto_np).float()

        return aggregated_parameters, aggregated_metrics