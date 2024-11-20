def freeze_bn(self):
    """Freeze BatchNorm layers."""
    for layer in self.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.eval()
