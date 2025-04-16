def forward(self, x, context=None):
    b, c, h, w = x.shape
    x_in = x
    x = self.norm(x)
    x = self.proj_in(x)
    x = rearrange(x, 'b c h w -> b (h w) c')
    for block in self.transformer_blocks:
        x = block(x, context=context)
    x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
    x = self.proj_out(x)
    return x + x_in
