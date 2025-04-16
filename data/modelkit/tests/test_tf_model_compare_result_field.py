def compare_result_field(x, y, tolerance):
    """compares the objects x and y, whether they be python immutable types,
    iterables or numerical arrays (within a given tolerance)
    """
    assert type(x) == type(y)
    if isinstance(x, np.ndarray):
        if np.issubdtype(x.dtype, np.number):
            return (_abs_difference(x, y) <= tolerance).all()
        return (x == y).all()
    if isinstance(x, (float, int, complex, bool)):
        return np.abs(x - y) <= tolerance
    return x == y
