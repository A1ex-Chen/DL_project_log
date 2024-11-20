def add_arrays(self, arrays):
    """
        Add :obj:`arrays` to the internal storage, Will initialize the storage to the full size at the first arrays
        passed so that if we're bound to get an OOM, it happens at the beginning.
        """
    if arrays is None:
        return
    if self._storage is None:
        self._storage = nested_new_like(arrays, self.total_samples,
            padding_index=self.padding_index)
        self._offsets = list(range(0, self.total_samples, self.process_length))
    else:
        storage_shape = _get_first_shape(self._storage)
        arrays_shape = _get_first_shape(arrays)
        if len(storage_shape) > 1 and storage_shape[1] < arrays_shape[1]:
            self._storage = nested_expand_like(self._storage, arrays_shape[
                1], padding_index=self.padding_index)
    slice_len = self._nested_set_tensors(self._storage, arrays)
    for i in range(self.world_size):
        self._offsets[i] += slice_len
