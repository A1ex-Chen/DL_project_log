def _get_first_shape(arrays):
    """Return the shape of the first array found in the nested struct `arrays`."""
    if isinstance(arrays, (list, tuple)):
        return _get_first_shape(arrays[0])
    return arrays.shape
