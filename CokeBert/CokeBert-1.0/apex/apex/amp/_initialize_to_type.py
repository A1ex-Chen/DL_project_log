def to_type(dtype, t):
    if isinstance(t, torch.Tensor):
        if not t.is_cuda:
            warnings.warn('An input tensor was not cuda.')
        if t.is_floating_point():
            return t.to(dtype)
        return t
    else:
        return t.to(dtype)
