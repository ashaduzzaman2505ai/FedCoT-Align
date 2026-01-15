#!/bin/bash

# 1. Test Sensitivity to Non-IID Skew (The "Heterogeneity" Test)
python experiments/ablation_study.py --ablation partitioning.skew_level --values 0.05 0.1 0.5

# 2. Test Impact of Alignment Loss (The "Ablation of Method" Test)
python experiments/ablation_study.py --ablation loss_weights.lambda_align --values 0.0 0.5 1.0 2.0