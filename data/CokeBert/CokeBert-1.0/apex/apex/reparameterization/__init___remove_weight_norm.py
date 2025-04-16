def remove_weight_norm(module, name='', remove_all=False):
    """
    Removes the weight normalization reparameterization of a parameter from a module.
    If no parameter is supplied then all weight norm parameterizations are removed.
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
    Example:
        >>> m = apply_weight_norm(nn.Linear(20, 40))
        >>> remove_weight_norm(m)
    """
    return remove_reparameterization(module, reparameterization=WeightNorm,
        name=name, remove_all=remove_all)
