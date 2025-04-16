def _convert_to_tensor(a, dtype=None, device=None):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if dtype is not None:
        a = a.type(dtype)
    if device is not None:
        a = a.to(device)
    return a
