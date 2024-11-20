def load_state_dict(self, state_dict):
    """Copies the state of the iterator from the given *state_dict*."""
    self.epoch = state_dict['epoch']
    itr_pos = state_dict.get('iterations_in_epoch', 0)
    if itr_pos > 0:
        itr = self._get_iterator_for_epoch(self.epoch, state_dict.get(
            'shuffle', True))
        if itr_pos < len(itr):
            self._next_epoch_itr = itr.skip(itr_pos)
