def _decreasing(t):
    return (t[1:] < t[:-1]).all()
