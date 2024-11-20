def fromarray(self, im):
    """Update self.im from a numpy array."""
    self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
    self.draw = ImageDraw.Draw(self.im)
