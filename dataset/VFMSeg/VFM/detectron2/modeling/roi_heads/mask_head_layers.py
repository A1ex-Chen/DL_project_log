def layers(self, x):
    for layer in self:
        x = layer(x)
    return x
