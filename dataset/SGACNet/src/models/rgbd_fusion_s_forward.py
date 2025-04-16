def forward(self, rgb, depth):
    rgb = self.se_rgb(rgb)
    depth = self.se_depth(depth)
    out = rgb + depth
    return out
