def sample(self, key):
    return self.mean + self.std * jax.random.normal(key, self.mean.shape)
