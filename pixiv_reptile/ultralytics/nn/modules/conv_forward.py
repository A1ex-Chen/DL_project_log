def forward(self, x):
    """Forward pass for the YOLOv8 mask Proto module."""
    return torch.cat(x, self.d)
