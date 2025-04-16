def forward(self, x):
    """Executes a forward pass on the input tensor through the constructed model layers."""
    return self.forward_features(x)
