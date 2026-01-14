import numpy as np
import torch
from typing import List, Dict, Any
from datasets import Dataset, concatenate_datasets

class NonIIDPartitioner:
    """
    Implements Dirichlet-based domain skew for Federated Learning.
    Each client receives a mixture of all datasets, but the distribution is 
    heavily skewed toward a 'primary' domain based on alpha.
    """
    def __init__(self, datasets: List[Dataset], num_clients: int, alpha: float = 0.5, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.datasets = datasets  # List of pre-processed Datasets
        self.num_clients = num_clients
        self.alpha = alpha
        self.num_domains = len(datasets)

    def partition(self) -> List[Dataset]:
        """
        Partitions the list of datasets across clients.
        Returns: List of Datasets (one per client).
        """
        # Matrix of shape (num_domains, num_clients)
        # Each row is a Dirichlet distribution of how one domain is split across clients
        dist = self.rng.dirichlet([self.alpha] * self.num_clients, self.num_domains)
        
        client_subsets = [[] for _ in range(self.num_clients)]

        for domain_idx, ds in enumerate(self.datasets):
            ds_size = len(ds)
            indices = self.rng.permutation(ds_size)
            
            # Calculate split points for this domain across clients
            split_points = (np.cumsum(dist[domain_idx]) * ds_size).astype(int)[:-1]
            client_indices = np.split(indices, split_points)
            
            for client_idx in range(self.num_clients):
                if len(client_indices[client_idx]) > 0:
                    client_subsets[client_idx].append(ds.select(client_indices[client_idx]))

        # Concatenate all domain-fragments for each client
        return [concatenate_datasets(subsets).shuffle(seed=42) for subsets in client_subsets]