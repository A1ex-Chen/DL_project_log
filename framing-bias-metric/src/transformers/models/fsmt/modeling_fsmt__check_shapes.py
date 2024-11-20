def _check_shapes(shape_1, shape2):
    if shape_1 != shape2:
        raise AssertionError('shape mismatch: {} != {}'.format(shape_1, shape2)
            )
