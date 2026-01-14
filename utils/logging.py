import logging
import sys
import wandb
from typing import Optional, Dict, Any

def setup_logging(use_wandb: bool = False, config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """Standardizes logging across clients and server."""
    logger = logging.getLogger("fedcot")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if use_wandb and wandb.run is None:
        wandb.init(
            project="fedcot-align",
            config=config,
            reinit=True,
            settings=wandb.Settings(start_method="fork")
        )
    
    return logger

def log_metrics(metrics: Dict[str, Any], step: int, prefix: str = "") -> None:
    logger = logging.getLogger("fedcot")
    log_str = f"Step {step} | " + " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
    logger.info(log_str)
    
    if wandb.run is not None:
        wandb_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        wandb.log(wandb_metrics, step=step)