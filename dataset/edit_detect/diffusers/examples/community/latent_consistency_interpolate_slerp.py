def slerp(v0: Union[torch.Tensor, np.ndarray], v1: Union[torch.Tensor, np.
    ndarray], t: Union[float, torch.Tensor, np.ndarray], DOT_THRESHOLD=0.9995
    ) ->Union[torch.Tensor, np.ndarray]:
    """
    Spherical linear interpolation between two vectors/tensors.

    Args:
        v0 (`torch.Tensor` or `np.ndarray`): First vector/tensor.
        v1 (`torch.Tensor` or `np.ndarray`): Second vector/tensor.
        t: (`float`, `torch.Tensor`, or `np.ndarray`):
            Interpolation factor. If float, must be between 0 and 1. If np.ndarray or
            torch.Tensor, must be one dimensional with values between 0 and 1.
        DOT_THRESHOLD (`float`, *optional*, default=0.9995):
            Threshold for when to use linear interpolation instead of spherical interpolation.

    Returns:
        `torch.Tensor` or `np.ndarray`:
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
        t = np.array([t], dtype=v0.dtype)
    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = lerp(v0, v1, t)
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        s0 = s0[..., None]
        s1 = s1[..., None]
        v0 = v0[None, ...]
        v1 = v1[None, ...]
        v2 = s0 * v0 + s1 * v1
    if t_is_float and v0.ndim > 1:
        assert v2.shape[0] == 1
        v2 = np.squeeze(v2, axis=0)
    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)
    return v2
