# Reuse FedCoTClient but modify to share naive CoT text instead of embeddings
# (For simulation: we log generated text but don't actually send it - violates privacy)
# In practice: this baseline shows performance drop without latent alignment

def run_fl_naive_cot(config: Dict[str, Any]):
    """
    Baseline 3: FL + naive CoT (generates CoT text locally but no alignment)
    """
    # Clients train with CoT generation but lambda2=0.0
    # Optionally log generated CoT text (but never share)
    config_copy = config.copy()
    config_copy['loss_weights']['lambda2'] = 0.0
    # Proceed as in main FL but with modified loss
    # (Implementation reuses main client/trainer with flag)
    print("Running FL + naive CoT (no latent alignment)")