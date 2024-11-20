def remove_reparameterization(module, reparameterization=Reparameterization,
    name='', remove_all=False):
    """
    Removes the given reparameterization of a parameter from a module.
    If no parameter is supplied then all reparameterizations are removed.
    Args:
        module (nn.Module): containing module
        reparameterization (Reparameterization): reparamaterization class to apply
        name (str, optional): name of weight parameter
        remove_all (bool, optional): if True, remove all reparamaterizations of given type. Default: False
    Example:
        >>> m = apply_reparameterization(nn.Linear(20, 40),WeightNorm)
        >>> remove_reparameterization(m)
    """
    if name != '' or remove_all:
        to_remove = []
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, reparameterization) and (hook.name == name or
                remove_all):
                hook.remove(module)
                to_remove.append(k)
        if len(to_remove) > 0:
            for k in to_remove:
                del module._forward_pre_hooks[k]
            return module
        if not remove_all:
            raise ValueError("reparameterization of '{}' not found in {}".
                format(name, module))
    else:
        modules = [module] + [x for x in module.modules()]
        for m in modules:
            remove_reparameterization(m, reparameterization=
                reparameterization, remove_all=True)
        return module
