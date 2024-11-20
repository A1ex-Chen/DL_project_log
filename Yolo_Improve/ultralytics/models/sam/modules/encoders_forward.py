def forward(self, x: torch.Tensor) ->torch.Tensor:
    """Computes patch embedding by applying convolution and transposing resulting tensor."""
    return self.proj(x).permute(0, 2, 3, 1)
