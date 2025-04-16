@property
def num_channels(self) ->int:
    """The number of image channels (can be inferred from the dataset)."""
    return int(self._num_channels)
