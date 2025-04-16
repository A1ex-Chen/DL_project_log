def resample_using_grid_sample():
    grid_x = torch.arange(0, out_size[1], 1, dtype=input.dtype, device=
        input.device)
    grid_x = grid_x * (2 * scale_x / (input.shape[3] - 1)) - 1
    grid_y = torch.arange(0, out_size[0], 1, dtype=input.dtype, device=
        input.device)
    grid_y = grid_y * (2 * scale_y / (input.shape[2] - 1)) - 1
    grid_x = grid_x.view(1, out_size[1]).repeat(out_size[0], 1)
    grid_y = grid_y.view(out_size[0], 1).repeat(1, out_size[1])
    grid_xy = torch.cat((grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)), dim=2
        ).unsqueeze(0)
    grid_xy = grid_xy.repeat(input.shape[0], 1, 1, 1)
    out = F.grid_sample(input, grid_xy, mode='bilinear', padding_mode=
        'border', align_corners=True)
    return out
