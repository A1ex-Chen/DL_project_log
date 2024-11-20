def forward(self, x: torch.Tensor):
    x = self.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0
        ], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
    x = x + self.positional_embedding.to(x.dtype)
    x = self.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = self.transformer(x)
    x = x.permute(1, 0, 2)
    return x
