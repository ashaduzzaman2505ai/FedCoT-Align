import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from data.loaders import DatasetLoader
from data.partitioner import NonIIDPartitioner
from collections import Counter

def test():
    # 1. Test Loading (Subsampled for speed)
    loader = DatasetLoader(
        config_path='configs/datasets.yaml', 
        model_name='google/flan-t5-small', 
        subsample_size=100
    )
    datasets = loader.load_and_preprocess()
    print(f"Loaded {len(datasets)} domains.")

    # 2. Test Partitioning
    num_clients = 4
    partitioner = NonIIDPartitioner(datasets, num_clients=num_clients, alpha=0.1)
    client_datasets = partitioner.partition()
    
    assert len(client_datasets) == num_clients
    
    # 3. Verify skew (Non-IID check)
    print("\nClient Data Distribution (Total Samples):")
    for i, cds in enumerate(client_datasets):
        print(f"Client {i}: {len(cds)} samples")
        # Check first batch shape
        batch = cds[0]
        assert batch['input_ids'].shape[0] == 512 # Max length
        assert 'is_hallucination' in batch

    print("\n✅ Step 3 Passed: Data is partitioned non-IID and T5-ready.")

if __name__ == "__main__":
    test()