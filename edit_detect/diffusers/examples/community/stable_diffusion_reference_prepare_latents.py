def prepare_latents(self, batch_size: int, num_channels_latents: int,
    height: int, width: int, dtype: torch.dtype, device: torch.device,
    generator: Union[torch.Generator, List[torch.Generator]], latents:
    Optional[torch.Tensor]=None) ->torch.Tensor:
    """
        Prepare the latent vectors for diffusion.

        Args:
            batch_size (int): The number of samples in the batch.
            num_channels_latents (int): The number of channels in the latent vectors.
            height (int): The height of the latent vectors.
            width (int): The width of the latent vectors.
            dtype (torch.dtype): The data type of the latent vectors.
            device (torch.device): The device to place the latent vectors on.
            generator (Union[torch.Generator, List[torch.Generator]]): The generator(s) to use for random number generation.
            latents (Optional[torch.Tensor]): The pre-existing latent vectors. If None, new latent vectors will be generated.

        Returns:
            torch.Tensor: The prepared latent vectors.
        """
    shape = batch_size, num_channels_latents, int(height
        ) // self.vae_scale_factor, int(width) // self.vae_scale_factor
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.'
            )
    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device,
            dtype=dtype)
    else:
        latents = latents.to(device)
    latents = latents * self.scheduler.init_noise_sigma
    return latents
