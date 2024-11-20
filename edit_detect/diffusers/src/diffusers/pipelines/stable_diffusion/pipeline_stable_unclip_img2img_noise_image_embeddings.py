def noise_image_embeddings(self, image_embeds: torch.Tensor, noise_level:
    int, noise: Optional[torch.Tensor]=None, generator: Optional[torch.
    Generator]=None):
    """
        Add noise to the image embeddings. The amount of noise is controlled by a `noise_level` input. A higher
        `noise_level` increases the variance in the final un-noised images.

        The noise is applied in two ways:
        1. A noise schedule is applied directly to the embeddings.
        2. A vector of sinusoidal time embeddings are appended to the output.

        In both cases, the amount of noise is controlled by the same `noise_level`.

        The embeddings are normalized before the noise is applied and un-normalized after the noise is applied.
        """
    if noise is None:
        noise = randn_tensor(image_embeds.shape, generator=generator,
            device=image_embeds.device, dtype=image_embeds.dtype)
    noise_level = torch.tensor([noise_level] * image_embeds.shape[0],
        device=image_embeds.device)
    self.image_normalizer.to(image_embeds.device)
    image_embeds = self.image_normalizer.scale(image_embeds)
    image_embeds = self.image_noising_scheduler.add_noise(image_embeds,
        timesteps=noise_level, noise=noise)
    image_embeds = self.image_normalizer.unscale(image_embeds)
    noise_level = get_timestep_embedding(timesteps=noise_level,
        embedding_dim=image_embeds.shape[-1], flip_sin_to_cos=True,
        downscale_freq_shift=0)
    noise_level = noise_level.to(image_embeds.dtype)
    image_embeds = torch.cat((image_embeds, noise_level), 1)
    return image_embeds
