def min_max_prune(module, name, amount):
    """Prune tensor according to min max game theory"""
    MinMaxPrune.apply(module, name, amount)
    return module
