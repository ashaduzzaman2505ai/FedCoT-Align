from typing import Dict, List, Tuple, Any
import flwr as fl
import torch
from torch.utils.data import DataLoader

from models.fedcot_model import FedCoTModel
from training.local_trainer import LocalTrainer
from utils.seeding import set_seed

class FedCoTClient(fl.client.NumPyClient):
    """
    Flower client for FedCoT-Align.
    Performs local training and returns model updates + local CoT prototype (mean embedding).
    """
    def __init__(
        self,
        client_id: int,
        model: FedCoTModel,
        trainloader: DataLoader,
        valloader: DataLoader,
        config: Dict[str, Any],
        global_prototype: torch.Tensor = None
    ):
        self.client_id = client_id
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        set_seed(config['fl']['seed'] + client_id)  # Client-specific seed offset

        self.trainer = LocalTrainer(
            config=config,
            model=model,
            dataloader=trainloader,
            global_prototype=global_prototype.to(self.device) if global_prototype is not None else torch.zeros(
                config['heads']['projector']['embed_dim'], device=self.device
            )
        )

    def get_parameters(self, config=None) -> List[torch.Tensor]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[torch.Tensor]):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[torch.Tensor], config: Dict[str, Any]
    ) -> Tuple[List[torch.Tensor], int, Dict]:
        self.set_parameters(parameters)

        # Local training
        metrics = self.trainer.train(local_epochs=self.config['fl']['local_epochs'])

        # Compute local CoT prototype (mean of client embeddings)
        local_prototype = self._compute_local_prototype()

        updated_parameters = self.get_parameters()
        num_examples = len(self.trainloader.dataset)

        return updated_parameters, num_examples, {
            "local_prototype": local_prototype.cpu().numpy(),  # Sent to server
            **metrics
        }

    def evaluate(
        self, parameters: List[torch.Tensor], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        # Simple evaluation (extend with full eval later)
        loss = 0.0  # Placeholder; use metrics.py later
        num_examples = len(self.valloader.dataset) if self.valloader else 0
        return loss, num_examples, {"accuracy": 0.0}  # Placeholder

    def _compute_local_prototype(self) -> torch.Tensor:
        """Average CoT embeddings over local data (one forward pass)."""
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for batch in self.trainloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                )
                embeddings.append(outputs['cot_embedding'])
        if embeddings:
            return torch.cat(embeddings, dim=0).mean(dim=0)
        return torch.zeros(self.config['heads']['projector']['embed_dim'], device=self.device)