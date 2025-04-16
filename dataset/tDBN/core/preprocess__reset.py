def _reset(self):
    if self._name is not None:
        print('reset', self._name)
    if self._shuffle:
        np.random.shuffle(self._indices)
    self._idx = 0
