def get_output_and_grid(self, reg_box, hsize, wsize, k, stride, dtype):
    grid_size = self.grid_sizes[k]
    if grid_size[0] != hsize or grid_size[1] != wsize or grid_size[2
        ] != stride:
        grid_size[0] = hsize
        grid_size[1] = wsize
        grid_size[2] = stride
        yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
        grid = torch.stack((xv, yv), 2).view(1, -1, 2).type(dtype).contiguous()
        self.grids[k] = grid
        xy_shift = (grid + 0.5) * stride
        self.xy_shifts[k] = xy_shift
        expanded_stride = torch.full((1, grid.shape[1], 1), stride, dtype=
            grid.dtype, device=grid.device)
        self.expanded_strides[k] = expanded_stride
        center_radius = self.center_radius * expanded_stride
        center_radius = center_radius.expand_as(xy_shift)
        center_lt = center_radius + xy_shift
        center_rb = center_radius - xy_shift
        center_ltrb = torch.cat([center_lt, center_rb], dim=-1)
        self.center_ltrbes[k] = center_ltrb
    xy_shift = self.xy_shifts[k]
    grid = self.grids[k]
    expanded_stride = self.expanded_strides[k]
    center_ltrb = self.center_ltrbes[k]
    half_wh = torch.exp(reg_box[..., 2:4]) * (stride / 2)
    reg_box[..., :2] = (reg_box[..., :2] + grid) * stride
    reg_box[..., 2:4] = reg_box[..., :2] + half_wh
    reg_box[..., :2] = reg_box[..., :2] - half_wh
    return reg_box, grid, xy_shift, expanded_stride, center_ltrb
