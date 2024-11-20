def __init_subclass__(cls) ->None:
    """Register subclasses as pytree nodes.

        This is necessary to synchronize gradients when using `torch.nn.parallel.DistributedDataParallel` with
        `static_graph=True` with modules that output `ModelOutput` subclasses.
        """
    if is_torch_available():
        import torch.utils._pytree
        if is_torch_version('<', '2.2'):
            torch.utils._pytree._register_pytree_node(cls, torch.utils.
                _pytree._dict_flatten, lambda values, context: cls(**torch.
                utils._pytree._dict_unflatten(values, context)))
        else:
            torch.utils._pytree.register_pytree_node(cls, torch.utils.
                _pytree._dict_flatten, lambda values, context: cls(**torch.
                utils._pytree._dict_unflatten(values, context)))
