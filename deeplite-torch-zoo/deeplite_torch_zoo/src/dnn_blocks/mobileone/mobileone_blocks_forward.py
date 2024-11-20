def forward(self, x: torch.Tensor) ->torch.Tensor:
    x = self.depthwise_conv(x)
    x = self.pointwise_conv(x)
    return x
