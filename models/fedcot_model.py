import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from typing import Dict, Any, Optional
from models.heads.verifier import VerifierHead
from models.heads.projector import CoTProjector

class FedCoTModel(nn.Module):
    def __init__(self, config: Dict[str, Any], use_lora: bool = True):
        super().__init__()
        self.model_name = config['model']['name']
        self.hidden_size = config['model']['d_model']
        
        # 1. Load Base Model
        if 't5' in self.model_name.lower():
            self.model_type = "seq2seq"
            self.base_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            target_modules = ["q", "v"]
        elif 'llama' in self.model_name.lower() or 'mistral' in self.model_name.lower():
            self.model_type = "causal"
            self.base_model = AutoModelForCausalLM.from_pretrained(self.model_name)
            target_modules = ["q_proj", "v_proj"]
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")

        # 2. Apply PEFT/LoRA
        if use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM if self.model_type == "seq2seq" else TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=config['model']['dropout'],
                target_modules=target_modules
            )
            self.base_model = get_peft_model(self.base_model, peft_config)
            print(f"PEFT/LoRA initialized for {self.model_name}")

        # 3. Modular Heads (These remain fully trainable)
        self.reasoning_head = nn.Linear(self.hidden_size, config['heads']['reasoning']['output_dim'])
        
        verifier_input_dim = self.hidden_size + config['heads']['reasoning']['output_dim']
        self.verifier_head = VerifierHead(
            input_dim=verifier_input_dim,
            hidden_dim=config['heads']['verifier']['hidden_dim'],
            num_classes=config['heads']['verifier']['num_classes']
        )
        
        self.cot_projector = CoTProjector(
            input_dim=config['heads']['reasoning']['output_dim'],
            embed_dim=config['heads']['projector']['embed_dim']
        )
        
        self.dropout = nn.Dropout(config['model']['dropout'])

    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        # Base model forward
        if self.model_type == "seq2seq":
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True
            )
            last_hidden = outputs.encoder_last_hidden_state
        else:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            # For Causal models, hidden_states is a tuple of all layers
            last_hidden = outputs.hidden_states[-1]

        # Mean pooling across sequence dimension
        pooled_hidden = last_hidden.mean(dim=1)
        
        # Reasoning & Alignment
        cot_states = self.reasoning_head(self.dropout(pooled_hidden))
        cot_embedding = self.cot_projector(cot_states)
        
        # Verifier Logits (Fused pooled hidden + CoT states)
        verifier_input = torch.cat([pooled_hidden, cot_states], dim=-1)
        verifier_logits = self.verifier_head(verifier_input)

        return {
            'loss': outputs.loss if hasattr(outputs, 'loss') else None,
            'answer_logits': outputs.logits,
            'cot_states': cot_states,
            'cot_embedding': cot_embedding,
            'verifier_logits': verifier_logits
        }