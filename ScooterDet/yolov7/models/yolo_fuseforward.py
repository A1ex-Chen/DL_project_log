def fuseforward(self, x):
    z = []
    self.training |= self.export
    for i in range(self.nl):
        x[i] = self.m[i](x[i])
        bs, _, ny, nx = x[i].shape
        x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2
            ).contiguous()
        if not self.training:
            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
            y = x[i].sigmoid()
            if not torch.onnx.is_in_onnx_export():
                y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]
                    ) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
            else:
                xy = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]
                wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].data
                y = torch.cat((xy, wh, y[..., 4:]), -1)
            z.append(y.view(bs, -1, self.no))
    if self.training:
        out = x
    elif self.end2end:
        out = torch.cat(z, 1)
    elif self.include_nms:
        z = self.convert(z)
        out = z,
    elif self.concat:
        out = torch.cat(z, 1)
    else:
        out = torch.cat(z, 1), x
    return out
