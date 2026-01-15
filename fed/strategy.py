import torch
import numpy as np
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict, Optional

class FedCoTAlignStrategy(FedAvg):
    def __init__(self, global_prototype_init: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.global_prototype = global_prototype_init

    def configure_fit(self, server_round, parameters, client_manager):
        # 1. Convert prototype to bytes
        proto_np = self.global_prototype.detach().cpu().numpy().astype(np.float32)
        proto_bytes = proto_np.tobytes() 
        
        # 2. Config must NOT contain lists. Convert shape to string.
        config = {
            "global_prototype_bytes": proto_bytes,
            "proto_shape_str": str(proto_np.shape), 
            "server_round": server_round
        }
        
        instructions = super().configure_fit(server_round, parameters, client_manager)
        for _, fit_ins in instructions:
            fit_ins.config.update(config)
        return instructions

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if not results:
            return aggregated_parameters, aggregated_metrics

        protos, weights = [], []
        for _, fit_res in results:
            # FIX: Metrics now contain bytes, not a numpy array/list
            if "local_prototype" in fit_res.metrics:
                raw_bytes = fit_res.metrics["local_prototype"]
                # Reconstruct numpy array from bytes
                proto_np = np.frombuffer(raw_bytes, dtype=np.float32).copy()
                protos.append(proto_np)
                weights.append(fit_res.num_examples)
        
        if protos:
            protos_np = np.array(protos)
            weights_np = np.array(weights).reshape(-1, 1)
            # Weighted average
            new_proto_np = np.sum(protos_np * weights_np, axis=0) / np.sum(weights_np)
            self.global_prototype = torch.from_numpy(new_proto_np).float()

        return aggregated_parameters, aggregated_metrics