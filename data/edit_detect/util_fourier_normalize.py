def normalize(x: Union[np.ndarray, torch.Tensor], vmin_in: float=None,
    vmax_in: float=None, vmin_out: float=0, vmax_out: float=1, eps: float=1e-05
    ) ->Union[np.ndarray, torch.Tensor]:
    if vmax_out == None and vmin_out == None:
        return x
    if isinstance(x, np.ndarray):
        if vmin_in == None:
            min_x = np.min(x)
        else:
            min_x = vmin_in
        if vmax_in == None:
            max_x = np.max(x)
        else:
            max_x = vmax_in
    elif isinstance(x, torch.Tensor):
        if vmin_in == None:
            min_x = torch.min(x)
        else:
            min_x = vmin_in
        if vmax_in == None:
            max_x = torch.max(x)
        else:
            max_x = vmax_in
    else:
        raise TypeError('x must be a torch.Tensor or a np.ndarray')
    if vmax_out == None:
        vmax_out = max_x
    if vmin_out == None:
        vmin_out = min_x
    return (x - min_x) / (max_x - min_x + eps) * (vmax_out - vmin_out
        ) + vmin_out
