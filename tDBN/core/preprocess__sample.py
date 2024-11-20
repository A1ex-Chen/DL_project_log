def _sample(self, num):
    if self._idx + num >= self._example_num:
        ret = self._indices[self._idx:].copy()
        self._reset()
    else:
        ret = self._indices[self._idx:self._idx + num]
        self._idx += num
    return ret
