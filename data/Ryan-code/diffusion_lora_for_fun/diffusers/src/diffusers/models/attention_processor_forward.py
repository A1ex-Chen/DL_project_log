def forward(self, f: torch.Tensor, zq: torch.Tensor) ->torch.Tensor:
    f_size = f.shape[-2:]
    zq = F.interpolate(zq, size=f_size, mode='nearest')
    norm_f = self.norm_layer(f)
    new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
    return new_f
