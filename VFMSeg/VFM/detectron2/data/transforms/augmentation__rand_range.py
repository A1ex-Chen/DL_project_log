def _rand_range(self, low=1.0, high=None, size=None):
    """
        Uniform float random number between low and high.
        """
    if high is None:
        low, high = 0, low
    if size is None:
        size = []
    return np.random.uniform(low, high, size)
