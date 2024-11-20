def _init_weights(self, m):
    """Initializes weights for linear layers and layer normalization in the given module."""
    if isinstance(m, nn.Linear):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
