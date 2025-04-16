def __call__(self, clip_input, params: dict=None):
    clip_input = jnp.transpose(clip_input, (0, 2, 3, 1))
    return self.module.apply({'params': params or self.params}, jnp.array(
        clip_input, dtype=jnp.float32), rngs={})
