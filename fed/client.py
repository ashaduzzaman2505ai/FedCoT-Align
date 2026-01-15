import flwr as fl
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from models.fedcot_model import FedCoTModel
from training.local_trainer import LocalTrainer

class FedCoTClient(fl.client.NumPyClient):
    def __init__(self, client_id, model, trainloader, valloader, config):
        self.client_id = client_id
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_parameters(self, config=None) -> List[np.ndarray]:
        # Return only the parameters that require gradients (efficient for LoRA)
        return [val.cpu().numpy() for name, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        
        # --- DECODE GLOBAL PROTOTYPE ---
        if "global_prototype_bytes" in config:
            # Reconstruct from bytes
            proto_bytes = config["global_prototype_bytes"]
            # We assume the dimension from our model config
            dim = self.config['heads']['projector']['embed_dim']
            proto_np = np.frombuffer(proto_bytes, dtype=np.float32).copy()
            global_proto = torch.from_numpy(proto_np).to(self.device)
        else:
            global_proto = torch.zeros(self.config['heads']['projector']['embed_dim'], device=self.device)
            
    
        # Initialize/Update Trainer with current global consensus
        trainer = LocalTrainer(self.config, self.model, self.trainloader, global_proto)
        
        # Train
        metrics = trainer.train(epochs=self.config['fl']['local_epochs'])
        
        # Compute Local Prototype
        local_proto = self._compute_local_prototype()

        return self.get_parameters(), len(self.trainloader.dataset), {
                "local_prototype": local_proto.cpu().numpy().tobytes(), # Send back as bytes
                **metrics
        }

    def _compute_local_prototype(self) -> torch.Tensor:
        self.model.eval()
        all_embeddings = []
        with torch.no_grad():
            for batch in self.trainloader:
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                out = self.model(batch['input_ids'], batch['attention_mask'])
                all_embeddings.append(out['cot_embedding'])
        return torch.cat(all_embeddings, dim=0).mean(dim=0).cpu()