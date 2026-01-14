import logging
import wandb
from typing import Optional, Dict

def setup_logging(use_wandb: bool = False, config: Optional[Dict] = None) -> None:
    """
    Setup logging with optional Weights & Biases integration.

    Args:
        use_wandb (bool): Flag to enable W&B logging.
        config (Optional[Dict]): Config dict to log to W&B.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    if use_wandb:
        wandb.init(project="fedcot-align", config=config)
        logger.info("W&B initialized.")
    else:
        logger.info("Logging to console only.")

def log_metrics(metrics: Dict[str, float], step: int, use_wandb: bool = False) -> None:
    """
    Log metrics to console and optionally to W&B.

    Args:
        metrics (Dict[str, float]): Metrics to log.
        step (int): Current step or epoch.
        use_wandb (bool): Flag to log to W&B.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Step {step}: {metrics}")
    if use_wandb:
        wandb.log(metrics, step=step)