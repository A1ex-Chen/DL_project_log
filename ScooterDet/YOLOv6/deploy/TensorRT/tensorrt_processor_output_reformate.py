def output_reformate(self, outputs):
    z = []
    for i in range(self.nl):
        cls_output = outputs[3 * i].reshape((1, -1, self.shape[i], self.
            shape[i]))
        reg_output = outputs[3 * i + 1].reshape((1, -1, self.shape[i], self
            .shape[i]))
        obj_output = outputs[3 * i + 2].reshape((1, -1, self.shape[i], self
            .shape[i]))
        y = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid
            ()], 1)
        bs, _, ny, nx = y.shape
        y = y.view(bs, -1, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        if self.grid[i].shape[2:4] != y.shape[2:4]:
            d = self.stride.device
            yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(
                nx).to(d)], indexing='ij')
            self.grid[i] = torch.stack((xv, yv), 2).view(1, self.na, ny, nx, 2
                ).float()
        if self.inplace:
            y[..., 0:2] = (y[..., 0:2] + self.grid[i]) * self.stride[i]
            y[..., 2:4] = torch.exp(y[..., 2:4]) * self.stride[i]
        else:
            xy = (y[..., 0:2] + self.grid[i]) * self.stride[i]
            wh = torch.exp(y[..., 2:4]) * self.stride[i]
            y = torch.cat((xy, wh, y[..., 4:]), -1)
        z.append(y.view(bs, -1, self.no))
    return torch.cat(z, 1)
