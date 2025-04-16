def init_weights(self, rng: jax.Array) ->FrozenDict:
    sample_shape = 1, self.in_channels, self.sample_size, self.sample_size
    sample = jnp.zeros(sample_shape, dtype=jnp.float32)
    timesteps = jnp.ones((1,), dtype=jnp.int32)
    encoder_hidden_states = jnp.zeros((1, 1, self.cross_attention_dim),
        dtype=jnp.float32)
    params_rng, dropout_rng = jax.random.split(rng)
    rngs = {'params': params_rng, 'dropout': dropout_rng}
    added_cond_kwargs = None
    if self.addition_embed_type == 'text_time':
        is_refiner = (5 * self.config.addition_time_embed_dim + self.config
            .cross_attention_dim == self.config.
            projection_class_embeddings_input_dim)
        num_micro_conditions = 5 if is_refiner else 6
        text_embeds_dim = (self.config.
            projection_class_embeddings_input_dim - num_micro_conditions *
            self.config.addition_time_embed_dim)
        time_ids_channels = (self.projection_class_embeddings_input_dim -
            text_embeds_dim)
        time_ids_dims = time_ids_channels // self.addition_time_embed_dim
        added_cond_kwargs = {'text_embeds': jnp.zeros((1, text_embeds_dim),
            dtype=jnp.float32), 'time_ids': jnp.zeros((1, time_ids_dims),
            dtype=jnp.float32)}
    return self.init(rngs, sample, timesteps, encoder_hidden_states,
        added_cond_kwargs)['params']
