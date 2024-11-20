def kl(self, other=None):
    if self.deterministic:
        return jnp.array([0.0])
    if other is None:
        return 0.5 * jnp.sum(self.mean ** 2 + self.var - 1.0 - self.logvar,
            axis=[1, 2, 3])
    return 0.5 * jnp.sum(jnp.square(self.mean - other.mean) / other.var + 
        self.var / other.var - 1.0 - self.logvar + other.logvar, axis=[1, 2, 3]
        )
