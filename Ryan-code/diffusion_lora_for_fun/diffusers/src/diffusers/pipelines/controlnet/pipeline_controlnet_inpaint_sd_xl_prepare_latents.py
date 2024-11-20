def prepare_latents(self, batch_size, num_channels_latents, height, width,
    dtype, device, generator, latents=None, image=None, timestep=None,
    is_strength_max=True, add_noise=True, return_noise=False,
    return_image_latents=False):
    shape = batch_size, num_channels_latents, int(height
        ) // self.vae_scale_factor, int(width) // self.vae_scale_factor
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.'
            )
    if (image is None or timestep is None) and not is_strength_max:
        raise ValueError(
            'Since strength < 1. initial latents are to be initialised as a combination of Image + Noise.However, either the image or the noise timestep has not been provided.'
            )
    if return_image_latents or latents is None and not is_strength_max:
        image = image.to(device=device, dtype=dtype)
        if image.shape[1] == 4:
            image_latents = image
        else:
            image_latents = self._encode_vae_image(image=image, generator=
                generator)
        image_latents = image_latents.repeat(batch_size // image_latents.
            shape[0], 1, 1, 1)
    if latents is None and add_noise:
        noise = randn_tensor(shape, generator=generator, device=device,
            dtype=dtype)
        latents = noise if is_strength_max else self.scheduler.add_noise(
            image_latents, noise, timestep)
        latents = (latents * self.scheduler.init_noise_sigma if
            is_strength_max else latents)
    elif add_noise:
        noise = latents.to(device)
        latents = noise * self.scheduler.init_noise_sigma
    else:
        noise = randn_tensor(shape, generator=generator, device=device,
            dtype=dtype)
        latents = image_latents.to(device)
    outputs = latents,
    if return_noise:
        outputs += noise,
    if return_image_latents:
        outputs += image_latents,
    return outputs
