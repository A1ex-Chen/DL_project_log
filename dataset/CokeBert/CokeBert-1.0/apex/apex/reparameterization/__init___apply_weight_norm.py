def apply_weight_norm(module, name='', dim=0, hook_child=True):
    """
    Applies weight normalization to a parameter in the given module.
    If no parameter is provided, applies weight normalization to all
    parameters in model (except 1-d vectors and scalars).

    .. math::
         \\mathbf{w} = g \\dfrac{\\mathbf{v}}{\\|\\mathbf{v}\\|}

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by `name` (e.g. "weight") with two parameters: one specifying the magnitude
    (e.g. "weight_g") and one specifying the direction (e.g. "weight_v").
    Weight normalization is implemented via a hook that recomputes the weight
    tensor from the magnitude and direction before every :meth:`~Module.forward`
    call.

    By default, with `dim=0`, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    `dim=None`.

    See https://arxiv.org/abs/1602.07868

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm
        hook_child (boolean, optional): adds reparameterization hook to direct parent of the 
            parameters. If False, it's added to `module` instead. Default: True

    Returns:
        The original module with the weight norm hook

    Example::

        >>> m = apply_weight_norm(nn.Linear(20, 40), name='weight')
        Linear (20 -> 40)
        >>> m.weight_g.size()
        torch.Size([40, 1])
        >>> m.weight_v.size()
        torch.Size([40, 20])

    """
    return apply_reparameterization(module, reparameterization=WeightNorm,
        hook_child=hook_child, name=name, dim=dim)
