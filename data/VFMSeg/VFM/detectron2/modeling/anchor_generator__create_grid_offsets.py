def _create_grid_offsets(size: List[int], stride: int, offset: float,
    device: torch.device):
    grid_height, grid_width = size
    shifts_x = torch.arange(offset * stride, grid_width * stride, step=
        stride, dtype=torch.float32, device=device)
    shifts_y = torch.arange(offset * stride, grid_height * stride, step=
        stride, dtype=torch.float32, device=device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return shift_x, shift_y
