def _lower_triangular_mask(shape):
    """Creates a lower-triangular boolean mask over the last 2 dimensions."""
    row_index = math_ops.cumsum(array_ops.ones(shape=shape, dtype=dtypes.
        int32), axis=-2)
    col_index = math_ops.cumsum(array_ops.ones(shape=shape, dtype=dtypes.
        int32), axis=-1)
    return math_ops.greater_equal(row_index, col_index)
