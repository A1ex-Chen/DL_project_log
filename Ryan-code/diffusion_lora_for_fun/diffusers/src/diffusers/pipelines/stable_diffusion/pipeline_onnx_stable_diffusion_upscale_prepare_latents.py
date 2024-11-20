def prepare_latents(self, batch_size, num_channels_latents, height, width,
    dtype, generator, latents=None):
    shape = batch_size, num_channels_latents, height, width
    if latents is None:
        latents = generator.randn(*shape).astype(dtype)
    elif latents.shape != shape:
        raise ValueError(
            f'Unexpected latents shape, got {latents.shape}, expected {shape}')
    return latents
