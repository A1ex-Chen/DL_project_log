def prepare_latents(self, batch_size, num_channels_latents, height, width,
    dtype, device, generator, latents=None):
    shape = batch_size, num_channels_latents, height // 8, width // 8
    if latents is None:
        if device.type == 'mps':
            latents = torch.randn(shape, generator=generator, device='cpu',
                dtype=dtype).to(device)
        else:
            latents = torch.randn(shape, generator=generator, device=device,
                dtype=dtype)
    else:
        if latents.shape != shape:
            raise ValueError(
                f'Unexpected latents shape, got {latents.shape}, expected {shape}'
                )
        latents = latents.to(device)
    return latents
