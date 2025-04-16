def forward(self, x):
    inv_depth = super().forward(x).squeeze(dim=1)
    if self.invert:
        depth = self.scale * inv_depth + self.shift
        depth[depth < 1e-08] = 1e-08
        depth = 1.0 / depth
        return depth
    else:
        return inv_depth
