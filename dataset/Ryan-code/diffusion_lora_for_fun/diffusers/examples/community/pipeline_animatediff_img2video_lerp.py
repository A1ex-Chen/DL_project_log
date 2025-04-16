def lerp(v0: torch.Tensor, v1: torch.Tensor, t: Union[float, torch.Tensor]
    ) ->torch.Tensor:
    """
    Linear Interpolation between two tensors.

    Args:
        v0 (`torch.Tensor`): First tensor.
        v1 (`torch.Tensor`): Second tensor.
        t: (`float` or `torch.Tensor`): Interpolation factor.
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
    t = t[..., None]
    v0 = v0[None, ...]
    v1 = v1[None, ...]
    v2 = (1 - t) * v0 + t * v1
    if t_is_float and v0.ndim > 1:
        assert v2.shape[0] == 1
        v2 = np.squeeze(v2, axis=0)
    v2 = torch.from_numpy(v2).to(input_device)
    return v2
