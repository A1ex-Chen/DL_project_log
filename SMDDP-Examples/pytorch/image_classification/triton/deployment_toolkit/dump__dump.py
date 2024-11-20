def _dump(self, prefix, data):
    idx = self._items_counters.setdefault(prefix, 0)
    filename = f'{prefix}-{idx:012d}.npz'
    output_path = self._output_dir / filename
    if self._compress:
        np.savez_compressed(output_path, **data)
    else:
        np.savez(output_path, **data)
    nitems = len(list(data.values())[0])
    msg_for_labels = (
        'If these are correct shapes - consider moving loading of them into metrics.py.'
         if prefix == 'labels' else '')
    shapes = {name: (value.shape if isinstance(value, np.ndarray) else (len
        (value),)) for name, value in data.items()}
    assert all(len(v) == nitems for v in data.values()
        ), f'All items in "{prefix}" shall have same size on 0 axis equal to batch size. {msg_for_labels}{\', \'.join(f\'{name}: {shape}\' for name, shape in shapes.items())}'
    self._items_counters[prefix] += nitems
