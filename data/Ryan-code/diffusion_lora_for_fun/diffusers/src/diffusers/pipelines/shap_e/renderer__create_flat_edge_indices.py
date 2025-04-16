def _create_flat_edge_indices(flat_cube_indices: torch.Tensor, grid_size:
    Tuple[int, int, int]):
    num_xs = (grid_size[0] - 1) * grid_size[1] * grid_size[2]
    y_offset = num_xs
    num_ys = grid_size[0] * (grid_size[1] - 1) * grid_size[2]
    z_offset = num_xs + num_ys
    return torch.stack([flat_cube_indices[:, 0] * grid_size[1] * grid_size[
        2] + flat_cube_indices[:, 1] * grid_size[2] + flat_cube_indices[:, 
        2], flat_cube_indices[:, 0] * grid_size[1] * grid_size[2] + (
        flat_cube_indices[:, 1] + 1) * grid_size[2] + flat_cube_indices[:, 
        2], flat_cube_indices[:, 0] * grid_size[1] * grid_size[2] + 
        flat_cube_indices[:, 1] * grid_size[2] + flat_cube_indices[:, 2] + 
        1, flat_cube_indices[:, 0] * grid_size[1] * grid_size[2] + (
        flat_cube_indices[:, 1] + 1) * grid_size[2] + flat_cube_indices[:, 
        2] + 1, y_offset + flat_cube_indices[:, 0] * (grid_size[1] - 1) *
        grid_size[2] + flat_cube_indices[:, 1] * grid_size[2] +
        flat_cube_indices[:, 2], y_offset + (flat_cube_indices[:, 0] + 1) *
        (grid_size[1] - 1) * grid_size[2] + flat_cube_indices[:, 1] *
        grid_size[2] + flat_cube_indices[:, 2], y_offset + 
        flat_cube_indices[:, 0] * (grid_size[1] - 1) * grid_size[2] + 
        flat_cube_indices[:, 1] * grid_size[2] + flat_cube_indices[:, 2] + 
        1, y_offset + (flat_cube_indices[:, 0] + 1) * (grid_size[1] - 1) *
        grid_size[2] + flat_cube_indices[:, 1] * grid_size[2] +
        flat_cube_indices[:, 2] + 1, z_offset + flat_cube_indices[:, 0] *
        grid_size[1] * (grid_size[2] - 1) + flat_cube_indices[:, 1] * (
        grid_size[2] - 1) + flat_cube_indices[:, 2], z_offset + (
        flat_cube_indices[:, 0] + 1) * grid_size[1] * (grid_size[2] - 1) + 
        flat_cube_indices[:, 1] * (grid_size[2] - 1) + flat_cube_indices[:,
        2], z_offset + flat_cube_indices[:, 0] * grid_size[1] * (grid_size[
        2] - 1) + (flat_cube_indices[:, 1] + 1) * (grid_size[2] - 1) +
        flat_cube_indices[:, 2], z_offset + (flat_cube_indices[:, 0] + 1) *
        grid_size[1] * (grid_size[2] - 1) + (flat_cube_indices[:, 1] + 1) *
        (grid_size[2] - 1) + flat_cube_indices[:, 2]], dim=-1)
