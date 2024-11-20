def forward(self, x):
    z = []
    for i in range(self.nl):
        x[i] = self.m[i](x[i])
        if not self.no_post_processing:
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3,
                4, 2).contiguous()
            if not self.training:
                if self.dynamic:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx,
                        ny, i)
                with no_jit_trace():
                    if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                        self.grid[i], self.anchor_grid[i] = self._make_grid(nx,
                            ny, i)
                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]
                wh = (wh * 2) ** 2 * self.anchor_grid[i]
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))
    if self.no_post_processing:
        return x
    return x if self.training else (torch.cat(z, 1),) if self.export else (
        torch.cat(z, 1), x)
