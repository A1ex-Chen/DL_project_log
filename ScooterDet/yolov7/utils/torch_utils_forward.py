def forward(self, x, augment=False, profile=False):
    out = self.model(x)
    out = self.detect_layer(out)
    return out
