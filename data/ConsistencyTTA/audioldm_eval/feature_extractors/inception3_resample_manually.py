def resample_manually():
    grid_x = torch.arange(0, out_size[1], 1, dtype=input.dtype, device=
        input.device)
    grid_x = grid_x * torch.tensor(scale_x, dtype=torch.float32)
    grid_x_lo = grid_x.long()
    grid_x_hi = (grid_x_lo + 1).clamp_max(input.shape[3] - 1)
    grid_dx = grid_x - grid_x_lo.float()
    grid_y = torch.arange(0, out_size[0], 1, dtype=input.dtype, device=
        input.device)
    grid_y = grid_y * torch.tensor(scale_y, dtype=torch.float32)
    grid_y_lo = grid_y.long()
    grid_y_hi = (grid_y_lo + 1).clamp_max(input.shape[2] - 1)
    grid_dy = grid_y - grid_y_lo.float()
    in_00 = input[:, :, grid_y_lo, :][:, :, :, grid_x_lo]
    in_01 = input[:, :, grid_y_lo, :][:, :, :, grid_x_hi]
    in_10 = input[:, :, grid_y_hi, :][:, :, :, grid_x_lo]
    in_11 = input[:, :, grid_y_hi, :][:, :, :, grid_x_hi]
    in_0 = in_00 + (in_01 - in_00) * grid_dx.view(1, 1, 1, out_size[1])
    in_1 = in_10 + (in_11 - in_10) * grid_dx.view(1, 1, 1, out_size[1])
    out = in_0 + (in_1 - in_0) * grid_dy.view(1, 1, out_size[0], 1)
    return out
