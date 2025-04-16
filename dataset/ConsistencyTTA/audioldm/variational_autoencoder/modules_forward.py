def forward(self, x):
    z_fs = self.encode_with_pretrained(x)
    z = self.proj_norm(z_fs)
    z = self.proj(z)
    z = nonlinearity(z)
    for submodel, downmodel in zip(self.model, self.downsampler):
        z = submodel(z, temb=None)
        z = downmodel(z)
    if self.do_reshape:
        z = rearrange(z, 'b c h w -> b (h w) c')
    return z
