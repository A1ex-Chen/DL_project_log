def forward(self, x: torch.Tensor) ->torch.Tensor:
    """Apply forward pass."""
    x = self.stage0(x)
    x = self.stage1(x)
    x = self.stage2(x)
    x = self.stage3(x)
    x = self.stage4(x)
    x = self.gap(x)
    x = x.view(x.size(0), -1)
    x = self.linear(x)
    return x
