def forward_split(self, x):
    """Forward pass using split() instead of chunk()."""
    y = list(self.cv1(x).split((self.c, self.c), 1))
    y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
    return self.cv4(torch.cat(y, 1))
