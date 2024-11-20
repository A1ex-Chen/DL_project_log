def init_weights(self, rng: jax.random.KeyArray) ->FrozenDict:
    sample_shape = 1, self.in_channels, self.sample_size, self.sample_size
    sample = jnp.zeros(sample_shape, dtype=jnp.float32)
    params_rng, dropout_rng, gaussian_rng = jax.random.split(rng, 3)
    rngs = {'params': params_rng, 'dropout': dropout_rng, 'gaussian':
        gaussian_rng}
    return self.init(rngs, sample)['params']
