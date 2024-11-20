def insertnoise(self, x, corruption_level=0.5):
    return np.random.binomial(1, 1 - corruption_level, x.shape) * x
