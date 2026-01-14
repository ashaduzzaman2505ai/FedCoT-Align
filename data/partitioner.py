import numpy as np
from typing import List, Dict, Any
from datasets import DatasetDict, Dataset
from scipy.stats import dirichlet

from utils.seeding import set_seed  # From Step 1

class NonIIDPartitioner:
    """
    Non-IID data partitioner for federated learning.
    Uses Dirichlet distribution for skew simulation.
    Each client gets data from primarily one domain, with configurable skew.

    Args:
        datasets (List[DatasetDict]): List of datasets (e.g., [gsm8k, truthfulqa, hotpotqa]).
        num_clients (int): Number of FL clients.
        skew_level (float): Dirichlet alpha; lower = more non-IID.
        seed (int): For reproducibility.
    """
    def __init__(self, datasets: List[DatasetDict], num_clients: int, skew_level: float = 0.5, seed: int = 42):
        set_seed(seed)
        self.datasets = datasets
        self.num_datasets = len(datasets)  # Number of domains (e.g., 3)
        self.num_clients = num_clients
        self.skew_level = skew_level
        if self.num_datasets > self.num_clients:
            raise ValueError("More datasets than clients; adjust config.")
        self.client_data: List[Dict[str, Dataset]] = [{} for _ in range(num_clients)]

    def partition(self) -> List[Dict[str, Dataset]]:
        """
        Partition datasets non-IID across clients.

        Returns:
            List[Dict[str, Dataset]]: Per-client data dicts, e.g., {'train': Dataset, 'test': Dataset}.
        """
        # Assign primary domains to clients (each client one main domain)
        domain_assignments = np.repeat(np.arange(self.num_datasets), self.num_clients // self.num_datasets)
        domain_assignments = np.append(domain_assignments, np.arange(self.num_clients % self.num_datasets))
        np.random.shuffle(domain_assignments)

        for client_id in range(self.num_clients):
            primary_domain = domain_assignments[client_id]
            proportions = dirichlet.rvs([self.skew_level] * self.num_datasets)[0]
            # Boost primary domain
            proportions[primary_domain] = max(proportions) + (1 - sum(proportions))

            for split in ['train', 'test']:  # Assume datasets have these splits
                client_split_data = []
                for ds_idx, ds in enumerate(self.datasets):
                    num_samples = int(len(ds[split]) * proportions[ds_idx])
                    sampled = ds[split].shuffle().select(range(num_samples))
                    client_split_data.append(sampled)
                # Concatenate sampled data from all domains, but skewed
                self.client_data[client_id][split] = Dataset.from_dict({
                    k: np.concatenate([d[k] for d in client_split_data]) for k in client_split_data[0].column_names
                }) if client_split_data else Dataset.from_dict({})

        return self.client_data

    def get_client_data(self, client_id: int) -> Dict[str, Dataset]:
        """
        Get data for a specific client.

        Args:
            client_id (int): Client index.

        Returns:
            Dict[str, Dataset]: Client's data splits.
        """
        return self.client_data[client_id]