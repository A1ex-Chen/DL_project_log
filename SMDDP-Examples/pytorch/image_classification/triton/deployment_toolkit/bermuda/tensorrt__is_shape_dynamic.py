def _is_shape_dynamic(input_shape):
    return any([(dim is None or dim == -1) for dim in input_shape])
