def nd_cicle(shape: Union[Tuple[int], int], diamiter: int):
    """
    Input:
    shape    : tuple (height, width)
    diameter : scalar
    
    Output:
    np.array of shape  that says True within a circle with diamiter =  around center 
    """
    if isinstance(shape, int):
        shape = shape, shape
    center: torch.Tensor = torch.tensor(shape) / 2.0
    idx_list: List[torch.Tensor] = []
    for d in shape:
        idx_list.append(torch.arange(d))
    grids: List[torch.Tensor] = torch.meshgrid(idx_list, indexing='ij')
    grid_residuals: List[torch.Tensor] = []
    for grid, c in zip(grids, center):
        grid_residuals.append((grid - c) ** 2)
    mask: torch.Tensor = torch.stack(grid_residuals, dim=0).sum(dim=0
        ) < diamiter ** 2
    return mask.int()
