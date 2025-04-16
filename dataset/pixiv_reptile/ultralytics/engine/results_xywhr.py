@property
def xywhr(self):
    """Return boxes in [x_center, y_center, width, height, rotation] format."""
    return self.data[:, :5]
