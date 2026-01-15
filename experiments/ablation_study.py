import yaml
import copy
import torch
import gc
import ray
from typing import Dict, Any, List
from experiments.main import run_fedcot_align

def set_nested_value(config: Dict, key_path: str, value: Any):
    """Update a nested dictionary using a dot-notated string (e.g., 'fl.num_rounds')."""
    keys = key_path.split('.')
    for key in keys[:-1]:
        config = config.setdefault(key, {})
    config[keys[-1]] = value

def run_ablation_suite(base_configs: List[str], ablation_key: str, values: List[Any]):
    """
    Iteratively runs experiments by varying a single parameter.
    """
    # 1. Load base configuration
    base_config = {}
    for path in base_configs:
        with open(path, 'r') as f:
            base_config.update(yaml.safe_load(f))

    for val in values:
        print(f"\n{'='*20}")
        print(f"STARTING ABLATION: {ablation_key} = {val}")
        print(f"{'='*20}\n")

        # 2. Create an isolated copy for this run
        current_config = copy.deepcopy(base_config)
        set_nested_value(current_config, ablation_key, val)
        
        # 3. Update W&B group/name for tracking
        if 'experiment' not in current_config:
            current_config['experiment'] = {}
        current_config['experiment']['run_name'] = f"ablate_{ablation_key}_{val}"

        try:
            # 4. Execute the FL simulation
            run_fedcot_align(current_config)
        except Exception as e:
            print(f"❌ Ablation run failed for {val}: {e}")
        finally:
            # 5. CRITICAL: Cleanup to prevent OOM in subsequent runs
            if ray.is_initialized():
                ray.shutdown()
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Automated Ablation Study Runner")
    parser.add_argument('--ablation', type=str, required=True, 
                        help="Dot-notated key: 'partitioning.skew_level' or 'loss_weights.lambda_align'")
    parser.add_argument('--values', nargs='+', type=float, required=True, 
                        help="List of numerical values to test")
    
    args = parser.parse_args()
    
    config_paths = ['configs/model.yaml', 'configs/fl.yaml', 'configs/datasets.yaml']
    run_ablation_suite(config_paths, args.ablation, args.values)