def shapes_to_tensor(x: List[int], device: Optional[torch.device]=None
    ) ->torch.Tensor:
    """
    Turn a list of integer scalars or integer Tensor scalars into a vector,
    in a way that's both traceable and scriptable.

    In tracing, `x` should be a list of scalar Tensor, so the output can trace to the inputs.
    In scripting or eager, `x` should be a list of int.
    """
    if torch.jit.is_scripting():
        return torch.as_tensor(x, device=device)
    if torch.jit.is_tracing():
        assert all([isinstance(t, torch.Tensor) for t in x]
            ), 'Shape should be tensor during tracing!'
        ret = torch.stack(x)
        if ret.device != device:
            ret = ret.to(device=device)
        return ret
    return torch.as_tensor(x, device=device)
