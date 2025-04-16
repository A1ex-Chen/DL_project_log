def lerp(v0: Union[torch.Tensor, np.ndarray], v1: Union[torch.Tensor, np.
    ndarray], t: Union[float, torch.Tensor, np.ndarray]) ->Union[torch.
    Tensor, np.ndarray]:
    """
    Linearly interpolate between two vectors/tensors.

    Args:
        v0 (`torch.Tensor` or `np.ndarray`): First vector/tensor.
        v1 (`torch.Tensor` or `np.ndarray`): Second vector/tensor.
        t: (`float`, `torch.Tensor`, or `np.ndarray`):
            Interpolation factor. If float, must be between 0 and 1. If np.ndarray or
            torch.Tensor, must be one dimensional with values between 0 and 1.

    Returns:
        Union[torch.Tensor, np.ndarray]
            Interpolated vector/tensor between v0 and v1.
    """
    inputs_are_torch = False
    t_is_float = False
    if isinstance(v0, torch.Tensor):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()
    if isinstance(t, torch.Tensor):
        inputs_are_torch = True
        input_device = t.device
        t = t.cpu().numpy()
    elif isinstance(t, float):
        t_is_float = True
        t = np.array([t])
    t = t[..., None]
    v0 = v0[None, ...]
    v1 = v1[None, ...]
    v2 = (1 - t) * v0 + t * v1
    if t_is_float and v0.ndim > 1:
        assert v2.shape[0] == 1
        v2 = np.squeeze(v2, axis=0)
    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)
    return v2
