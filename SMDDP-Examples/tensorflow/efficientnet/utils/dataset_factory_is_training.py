@property
def is_training(self) ->bool:
    """Whether this is the training set."""
    return self._split == 'train'
