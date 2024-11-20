def _merge_masks(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return math_ops.logical_and(x, y)
