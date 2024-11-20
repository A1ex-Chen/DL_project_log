def forward(self, x: torch.Tensor) ->torch.Tensor:
    x = self.forward_features(x)
    x = self.forward_head(x)
    return x
