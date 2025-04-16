def init_weights(self, rng: jax.Array, input_shape: Tuple, params:
    FrozenDict=None) ->FrozenDict:
    clip_input = jax.random.normal(rng, input_shape)
    params_rng, dropout_rng = jax.random.split(rng)
    rngs = {'params': params_rng, 'dropout': dropout_rng}
    random_params = self.module.init(rngs, clip_input)['params']
    return random_params
