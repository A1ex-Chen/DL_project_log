def __init__(self, parameters, deterministic=False):
    self.mean, self.logvar = jnp.split(parameters, 2, axis=-1)
    self.logvar = jnp.clip(self.logvar, -30.0, 20.0)
    self.deterministic = deterministic
    self.std = jnp.exp(0.5 * self.logvar)
    self.var = jnp.exp(self.logvar)
    if self.deterministic:
        self.var = self.std = jnp.zeros_like(self.mean)
