def compare_result(x, y, tolerance):
    """compares the objects x and y, whether they be python immutable types,
    iterables or numerical arrays (within a given tolerance)
    """
    assert type(x) == type(y)
    if isinstance(x, dict):
        assert set(x.keys()) == set(y.keys())
        for key in x:
            assert compare_result_field(x[key], y[key], tolerance)
        return True
    if isinstance(x, tuple):
        for xx, yy in zip(x, y):
            assert compare_result_field(xx, yy, tolerance)
        return True
    return compare_result_field(x, y, tolerance)
