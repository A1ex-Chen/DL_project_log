def _append_to_cache(self, prefix, data):
    if data is None:
        return
    if not isinstance(data, dict):
        raise ValueError(f'{prefix} data to store shall be dict')
    cached_data = self._items_cache.get(prefix, {})
    for name, value in data.items():
        assert isinstance(value, (list, np.ndarray)
            ), f'Values shall be lists or np.ndarrays; current type {type(value)}'
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        assert value.dtype.kind in ['S', 'U'] or not np.any(np.isnan(value)
            ), f'Values with np.nan is not supported; {name}={value}'
        cached_value = cached_data.get(name, None)
        if cached_value is not None:
            target_shape = np.max([cached_value.shape, value.shape], axis=0)
            cached_value = pad_except_batch_axis(cached_value, target_shape)
            value = pad_except_batch_axis(value, target_shape)
            value = np.concatenate((cached_value, value))
        cached_data[name] = value
    self._items_cache[prefix] = cached_data
