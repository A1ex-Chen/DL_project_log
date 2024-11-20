def auto_make_grids(samples: torch.Tensor):
    """
    Input/Output: Channel first
    """
    sample_grids = []
    for i in range(len(samples)):
        sample_grids.append(auto_make_grid(samples[i]))
    sample_grids = torch.stack(sample_grids)
    return sample_grids
