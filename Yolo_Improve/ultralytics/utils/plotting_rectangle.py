def rectangle(self, xy, fill=None, outline=None, width=1):
    """Add rectangle to image (PIL-only)."""
    self.draw.rectangle(xy, fill, outline, width)
