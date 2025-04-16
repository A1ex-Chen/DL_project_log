def _possibly_nonzero(x):
    return isinstance(x, torch.Tensor) or x != 0
