def init_weights(self, rng: jax.random.KeyArray) ->FrozenDict:
    sample_shape = 1, self.in_channels, self.sample_size, self.sample_size
    sample = jnp.zeros(sample_shape, dtype=jnp.float32)
    timesteps = jnp.ones((1,), dtype=jnp.int32)
    encoder_hidden_states = jnp.zeros((1, 1, self.cross_attention_dim),
        dtype=jnp.float32)
    params_rng, dropout_rng = jax.random.split(rng)
    rngs = {'params': params_rng, 'dropout': dropout_rng}
    return self.init(rngs, sample, timesteps, encoder_hidden_states)['params']
