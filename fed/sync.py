"""
Helper utilities for synchronization in FedCoT-Align.
Mainly for prototype distribution (model weights already handled by Flower).
"""
import torch

def broadcast_prototype(prototype: torch.Tensor, clients):
    """In simulation, just return; in real deployment, use gRPC or similar."""
    # For now, strategy holds it and clients receive via fit config if needed
    pass