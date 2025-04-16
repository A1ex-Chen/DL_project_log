@property
def num_classes(self) ->int:
    """The number of classes (can be inferred from the dataset)."""
    return int(self._num_classes)
