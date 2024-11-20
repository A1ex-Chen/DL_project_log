def forward(self, x: torch.Tensor):
    x = x.to(dtype=self.transformer.get_cast_dtype(), device=self.
        transformer.get_cast_device())
    x = self.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    x = x + get_abs_pos(self.positional_embedding, x.size(1))
    x = self.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = self.transformer(x)
    x = x.permute(1, 0, 2)
    x = self.attn_pool(x)
    x = self.ln_post(x)
    x = x @ self.proj
    return x
