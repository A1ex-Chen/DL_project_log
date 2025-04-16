@property
def image_size(self) ->int:
    """The size of each image (can be inferred from the dataset)."""
    return int(self._image_size)
