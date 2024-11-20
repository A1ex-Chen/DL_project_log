@property
def dummy_input(self):
    batch_size = 4
    num_channels = 3
    sizes = 32, 32
    prng_key = jax.random.PRNGKey(0)
    image = jax.random.uniform(prng_key, (batch_size, num_channels) + sizes)
    return {'sample': image, 'prng_key': prng_key}
