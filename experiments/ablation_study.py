import yaml
import copy
from typing import Dict, Any, List
from experiments.main import load_and_merge_config, run_fedcot_align
from utils.logging import log_metrics

def run_ablation(base_config_paths: List[str], ablation_key: str, values: List[Any]):
    """
    Run ablation by varying one config parameter (e.g., skew_level, lambda2).
    """
    base_config = load_and_merge_config(base_config_paths)

    for val in values:
        config = copy.deepcopy(base_config)
        keys = ablation_key.split('.')
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = val

        print(f"Running ablation: {ablation_key} = {val}")
        run_fedcot_align(config)
        log_metrics({f"{ablation_key}": val, "ablation_run": True}, step=0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', type=str, required=True,
                        help='Key to ablate, e.g., partitioning.skew_level or loss_weights.lambda2')
    parser.add_argument('--values', nargs='+', type=float, required=True,
                        help='Values to try')
    args = parser.parse_args()

    base_paths = ['configs/model.yaml', 'configs/fl.yaml', 'configs/datasets.yaml']
    run_ablation(base_paths, args.ablation, args.values)