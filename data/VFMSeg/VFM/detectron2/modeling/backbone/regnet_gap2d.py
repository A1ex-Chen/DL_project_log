def gap2d():
    """Helper for building a global average pooling layer."""
    return nn.AdaptiveAvgPool2d((1, 1))
