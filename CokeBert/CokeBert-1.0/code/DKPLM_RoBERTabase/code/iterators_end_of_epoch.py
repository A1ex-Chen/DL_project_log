def end_of_epoch(self):
    """Returns whether the most recent epoch iterator has been exhausted"""
    return not self._cur_epoch_itr.has_next()
