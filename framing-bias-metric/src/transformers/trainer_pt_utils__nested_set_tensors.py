def _nested_set_tensors(self, storage, arrays):
    if isinstance(arrays, (list, tuple)):
        for x, y in zip(storage, arrays):
            slice_len = self._nested_set_tensors(x, y)
        return slice_len
    assert arrays.shape[0
        ] % self.world_size == 0, f'Arrays passed should all have a first dimension multiple of {self.world_size}, found {arrays.shape[0]}.'
    slice_len = arrays.shape[0] // self.world_size
    for i in range(self.world_size):
        if len(arrays.shape) == 1:
            storage[self._offsets[i]:self._offsets[i] + slice_len] = arrays[
                i * slice_len:(i + 1) * slice_len]
        else:
            storage[self._offsets[i]:self._offsets[i] + slice_len, :arrays.
                shape[1]] = arrays[i * slice_len:(i + 1) * slice_len]
    return slice_len
