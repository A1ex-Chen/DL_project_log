def layers(self, x):
    for layer in self:
        x = layer(x)
    x = interpolate(x, scale_factor=self.up_scale, mode='bilinear',
        align_corners=False)
    return x
