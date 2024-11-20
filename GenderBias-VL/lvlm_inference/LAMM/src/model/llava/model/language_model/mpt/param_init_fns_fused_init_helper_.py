def fused_init_helper_(module: nn.Module, init_fn_):
    _fused = getattr(module, '_fused', None)
    if _fused is None:
        raise RuntimeError(f'Internal logic error')
    dim, splits = _fused
    splits = 0, *splits, module.weight.size(dim)
    for s, e in zip(splits[:-1], splits[1:]):
        slice_indices = [slice(None)] * module.weight.ndim
        slice_indices[dim] = slice(s, e)
        init_fn_(module.weight[slice_indices])
