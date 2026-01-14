#!/bin/bash
# Quick T4 test
python experiments/main.py --config configs/model.yaml configs/fl.yaml configs/datasets.yaml \
    --experiment main_results

# Full A100 run (edit subsample_ratio=1.0, batch_size=32, num_rounds=100 in YAML)
# python experiments/main.py ...