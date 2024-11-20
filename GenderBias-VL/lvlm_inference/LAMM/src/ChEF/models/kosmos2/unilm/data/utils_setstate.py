def setstate(self, checkpoint):
    if checkpoint is None:
        for iterator in self._source_iterators:
            iterator.setstate(None)
    else:
        for iterator, state in zip(self._source_iterators, checkpoint[
            'input_states']):
            iterator.setstate(state)
