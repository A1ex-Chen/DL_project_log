def __call__(self, i, bgr=False):
    """Converts hex color codes to RGB values."""
    c = self.palette[int(i) % self.n]
    return (c[2], c[1], c[0]) if bgr else c
