def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.
    __version__, '1.10.0')):
    d = self.anchors[i].device
    t = self.anchors[i].dtype
    shape = 1, self.na, ny, nx, 2
    y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d,
        dtype=t)
    yv, xv = torch.meshgrid(y, x, indexing='ij'
        ) if torch_1_10 else torch.meshgrid(y, x)
    grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
    anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)
        ).expand(shape)
    return grid, anchor_grid
