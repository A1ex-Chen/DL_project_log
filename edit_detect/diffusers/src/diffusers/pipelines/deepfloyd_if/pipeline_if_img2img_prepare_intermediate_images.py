def prepare_intermediate_images(self, image, timestep, batch_size,
    num_images_per_prompt, dtype, device, generator=None):
    _, channels, height, width = image.shape
    batch_size = batch_size * num_images_per_prompt
    shape = batch_size, channels, height, width
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.'
            )
    noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype
        )
    image = image.repeat_interleave(num_images_per_prompt, dim=0)
    image = self.scheduler.add_noise(image, noise, timestep)
    return image
