def slerp(v0: torch.Tensor, v1: torch.Tensor, t: Union[float, torch.Tensor],
    DOT_THRESHOLD: float=0.9995) ->torch.Tensor:
    """
    Spherical Linear Interpolation between two tensors.

    Args:
        v0 (`torch.Tensor`): First tensor.
        v1 (`torch.Tensor`): Second tensor.
        t: (`float` or `torch.Tensor`): Interpolation factor.
        DOT_THRESHOLD (`float`):
            Dot product threshold exceeding which linear interpolation will be used
            because input tensors are close to parallel.
    """
    t_is_float = False
    input_device = v0.device
    v0 = v0.cpu().numpy()
    v1 = v1.cpu().numpy()
    if isinstance(t, torch.Tensor):
        t = t.cpu().numpy()
    else:
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
    v2 = torch.from_numpy(v2).to(input_device)
    return v2
