def is_tensor(x) ->bool:
    """
    Tests if `x` is a `torch.Tensor` or `np.ndarray`.
    """
    if is_torch_available():
        import torch
        if isinstance(x, torch.Tensor):
            return True
    return isinstance(x, np.ndarray)
