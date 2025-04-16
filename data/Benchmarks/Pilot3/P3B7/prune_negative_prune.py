def negative_prune(module, name, amount):
    """Prune negative tensors"""
    NegativePrune.apply(module, name, amount)
    return module
