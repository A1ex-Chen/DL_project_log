def apply_reparameterization(module, reparameterization=None, name='', dim=
    0, hook_child=True):
    """
    Applies a given weight reparameterization (such as weight normalization) to
    a parameter in the given module. If no parameter is given, applies the reparameterization
    to all parameters in model (except 1-d vectors and scalars).

    Args:
        module (nn.Module): containing module
        reparameterization (Reparameterization): reparamaterization class to apply
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to perform reparameterization op
        hook_child (boolean, optional): adds reparameterization hook to direct parent of the 
            parameters. If False, it's added to `module` instead. Default: True

    Returns:
        The original module with the reparameterization hook

    Example::

        >>> m = apply_reparameterization(nn.Linear(20, 40), WeightNorm)
        Linear (20 -> 40)

    """
    assert reparameterization is not None
    if name != '':
        Reparameterization.apply(module, name, dim, reparameterization,
            hook_child)
    else:
        names = list(module.state_dict().keys())
        for name in names:
            apply_reparameterization(module, reparameterization, name, dim,
                hook_child)
    return module
