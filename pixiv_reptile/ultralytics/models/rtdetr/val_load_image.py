def load_image(self, i, rect_mode=False):
    """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
    return super().load_image(i=i, rect_mode=rect_mode)
