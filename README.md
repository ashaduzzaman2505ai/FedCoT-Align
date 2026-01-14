# FedCoT-Align
FedCoT-Align: Federated Alignment of Chain-of-Thought Reasoning for Hallucination Reduction

# Project Structure
```text
fedcot-align/
├── configs/
│ ├── experiments/ # Specific YAMLs for Table 1, Table 2, Ablations
│ ├── model.yaml
│ ├── fl.yaml
│ └── datasets.yaml
├── data/
│ ├── partitioner.py # MANDATORY: Logic for Non-IID Skew/Dirichlet
│ └── loaders.py # Dataset-specific tokenization/formatting
├── fed/
│ ├── client.py
│ ├── server.py
│ ├── strategy.py # Custom FedCoT-Align strategy (Prototype aggregation)
│ └── sync.py # Logic for handling embedding/gradient exchanges
├── models/
│ ├── fedcot_model.py # Wrapper for T5/LLaMA
│ ├── heads/ # Sub-modules for Verifier and Projector
│ │ ├── verifier.py
│ │ └── projector.py
├── training/
│ ├── local_trainer.py # Local loop (L_total = L_ans + L_ver + L_align)
│ └── losses.py # Implementation of tripartite loss
├── evaluation/
│ ├── metrics.py # EM, ROUGE, AUROC
│ ├── factuality.py # FACTScore / NLI implementations
│ └── consistency.py # Self-consistency variance logic
├── baselines/ # Wrappers for FedAvg, FedProx, Centralized
├── experiments/
│ ├── main.py # Unified entry point
│ └── ablation_study.py
├── scripts/ # Orchestration bash scripts
├── utils/ # Logging (W&B), Seeding, Checkpointing
├── pyproject.toml # Modern dependency management
└── README.md # Abstract, Method Diagram, Reproducibility
```

