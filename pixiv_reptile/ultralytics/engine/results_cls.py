@property
def cls(self):
    """Returns the class values of the oriented bounding boxes."""
    return self.data[:, -1]
