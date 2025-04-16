def _abs_difference(x, y):
    """a measure of the relative difference between two numbers."""
    return np.abs(x - y) / (0.0001 + (np.abs(x) + np.abs(y)) / 2)
