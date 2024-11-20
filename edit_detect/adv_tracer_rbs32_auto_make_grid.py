def auto_make_grid(sample: torch.Tensor, vmin_out: Union[float, int]=0,
    vmax_out: Union[float, int]=1):
    """
    Input/Output: Channel first
    """
    nrow = ceil(sqrt(len(sample)))
    return utils.make_grid(sample, nrow=nrow)
