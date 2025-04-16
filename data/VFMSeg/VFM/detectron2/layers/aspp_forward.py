def forward(self, x):
    size = x.shape[-2:]
    if self.pool_kernel_size is not None:
        if size[0] % self.pool_kernel_size[0] or size[1
            ] % self.pool_kernel_size[1]:
            raise ValueError(
                '`pool_kernel_size` must be divisible by the shape of inputs. Input size: {} `pool_kernel_size`: {}'
                .format(size, self.pool_kernel_size))
    res = []
    for conv in self.convs:
        res.append(conv(x))
    res[-1] = F.interpolate(res[-1], size=size, mode='bilinear',
        align_corners=False)
    res = torch.cat(res, dim=1)
    res = self.project(res)
    res = F.dropout(res, self.dropout, training=self.training
        ) if self.dropout > 0 else res
    return res
