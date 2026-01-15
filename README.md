# FedCoT-Align: Federated Latent Alignment for Reasoning Hallucination
[Under Development]

This repository contains the official implementation of **FedCoT-Align**, a federated learning framework designed to reduce hallucinations in Chain-of-Thought (CoT) reasoning through latent prototype alignment and hallucination verification.

## 🚀 Overview
FedCoT-Align addresses the challenge of reasoning drift in non-IID federated environments. It utilizes a **Tripartite Loss** objective:
1.  **Task Loss**: Standard cross-entropy for sequence generation.
2.  **Verifier Loss**: A supervised signal to detect internal hallucination states.
3.  **Latent Alignment**: A federated objective that aligns local client reasoning prototypes with a global consensus.



---

## 🛠 Installation

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/your-username/FedCoT-Align.git](https://github.com/your-username/FedCoT-Align.git)
    cd FedCoT-Align
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Core requirements: `flwr`, `torch`, `transformers`, `peft`, `ray`, `evaluate`.*

---

## 📊 Running Experiments

### 1. Main Experiment
To run the standard FedCoT-Align simulation (defaults to `google/flan-t5-small` with LoRA):
```bash
python experiments/main.py --config configs/model.yaml configs/fl.yaml configs/datasets.yaml

```

### 2. Baselines

Run the centralized "Upper Bound" or standard FedAvg comparisons:

```bash
# Centralized CoT (Non-federated)
python experiments/main.py --baselines --config configs/model.yaml configs/fl.yaml

# Standard FedAvg (Set lambda_align to 0.0)
python experiments/ablation_study.py --ablation loss_weights.lambda_align --values 0.0

```

### 3. Ablation Studies

Reproduce the sensitivity analysis for data heterogeneity (Dirichlet alpha):

```bash
python experiments/ablation_study.py --ablation partitioning.skew_level --values 0.01 0.1 0.5 1.0

```

---

## 📂 Project Structure

* `models/`: Core `FedCoTModel` and LoRA adapters.
* `training/`: `LocalTrainer` and Tripartite loss implementations.
* `fed/`: Flower client/server logic and custom `FedCoTAlignStrategy`.
* `evaluation/`: FactScore (NLI-based), ROUGE, and Exact Match metrics.
* `data/`: Non-IID partitioners and dataset loaders.
* `configs/`: YAML-based hyperparameter management.

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

---

## 📈 Evaluation Metrics

The framework automatically logs the following to W&B:

* **Exact Match (EM) & ROUGE**: Answer quality.
* **Verifier AUROC**: Ability of the model to self-detect hallucinations.
* **FactScore**: Percentage of generated atomic facts supported by a reference (via NLI).
* **Consistency Variance**: Self-consistency across multiple reasoning paths.

---

## 📜 Citation

If you use this code in your research, please cite:

```bibtex
@article{fedcot2026,
  title={FedCoT-Align: Federated Latent Alignment for Reasoning Hallucination},
  author={Your Name and Gemini AI},
  journal={arXiv preprint},
  year={2026}
}

```

