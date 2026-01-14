import os
from pathlib import Path

def setup_research_structure():
    """
    Initializes the FedCoT-Align project structure within the current directory.
    Designed for ICML-level reproducibility and modularity.
    """
    # Use current directory as project root
    root = Path(".")

    # 1. Define folder hierarchy
    folders = [
        "configs/experiments",
        "data",
        "fed",
        "models/heads",
        "training",
        "evaluation",
        "baselines",
        "experiments",
        "scripts",
        "utils",
    ]

    # 2. Define mandatory files
    files = [
        "configs/model.yaml",
        "configs/fl.yaml",
        "configs/datasets.yaml",
        "data/partitioner.py",
        "data/loaders.py",
        "fed/client.py",
        "fed/server.py",
        "fed/strategy.py",
        "fed/sync.py",
        "models/fedcot_model.py",
        "models/heads/verifier.py",
        "models/heads/projector.py",
        "training/local_trainer.py",
        "training/losses.py",
        "evaluation/metrics.py",
        "evaluation/factuality.py",
        "evaluation/consistency.py",
        "experiments/main.py",
        "experiments/ablation_study.py",
        "utils/logging.py",
        "utils/seed.py",
        "pyproject.toml",
        "requirements.txt",
        "README.md",
        ".gitignore"
    ]

    print(f"🛠️  Initializing FedCoT-Align structure in: {root.absolute()}")

    # Create directories and __init__.py files
    for folder in folders:
        path = root / folder
        path.mkdir(parents=True, exist_ok=True)
        
        # Add __init__.py to make it a package (skip for configs and scripts)
        if not any(x in folder for x in ["configs", "scripts"]):
            (path / "__init__.py").touch(exist_ok=True)
    
    # Create root __init__.py
    (root / "__init__.py").touch(exist_ok=True)

    # Create empty files if they don't exist
    for file_path in files:
        full_path = root / file_path
        if not full_path.exists():
            full_path.touch()
            print(f"  [CREATED] {file_path}")
        else:
            print(f"  [SKIP]    {file_path} (already exists)")


    print("\n✅ Project structure is ready.")
    print("Next suggested step: Define your model architecture in models/fedcot_model.py")

if __name__ == "__main__":
    setup_research_structure()