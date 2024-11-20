@property
def iterations_in_epoch(self):
    """The number of consumed batches in the current epoch."""
    if self._cur_epoch_itr is not None:
        return self._cur_epoch_itr.count
    elif self._next_epoch_itr is not None:
        return self._next_epoch_itr.count
    return 0
