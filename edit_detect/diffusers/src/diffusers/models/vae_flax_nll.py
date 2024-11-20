def nll(self, sample, axis=[1, 2, 3]):
    if self.deterministic:
        return jnp.array([0.0])
    logtwopi = jnp.log(2.0 * jnp.pi)
    return 0.5 * jnp.sum(logtwopi + self.logvar + jnp.square(sample - self.
        mean) / self.var, axis=axis)
