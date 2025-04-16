def volume_query_points(volume, grid_size):
    indices = torch.arange(grid_size ** 3, device=volume.bbox_min.device)
    zs = indices % grid_size
    ys = torch.div(indices, grid_size, rounding_mode='trunc') % grid_size
    xs = torch.div(indices, grid_size ** 2, rounding_mode='trunc') % grid_size
    combined = torch.stack([xs, ys, zs], dim=1)
    return combined.float() / (grid_size - 1) * (volume.bbox_max - volume.
        bbox_min) + volume.bbox_min
