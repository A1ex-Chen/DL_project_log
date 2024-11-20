def forward(self, x: torch.Tensor) ->torch.Tensor:
    x = self.proj(x)
    return x * torch.sigmoid(1.702 * x)
