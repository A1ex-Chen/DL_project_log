@torch.jit.ignore
def no_weight_decay_keywords(self):
    """Returns a dictionary of parameter names where weight decay should not be applied."""
    return {'attention_biases'}
