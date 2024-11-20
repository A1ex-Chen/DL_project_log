@torch.no_grad()
def single_infer(self, rgb_in: torch.Tensor, num_inference_steps: int, seed:
    Union[int, None], show_pbar: bool) ->torch.Tensor:
    """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
        Returns:
            `torch.Tensor`: Predicted depth map.
        """
    device = rgb_in.device
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps
    rgb_latent = self.encode_rgb(rgb_in)
    if seed is None:
        rand_num_generator = None
    else:
        rand_num_generator = torch.Generator(device=device)
        rand_num_generator.manual_seed(seed)
    depth_latent = torch.randn(rgb_latent.shape, device=device, dtype=self.
        dtype, generator=rand_num_generator)
    if self.empty_text_embed is None:
        self._encode_empty_text()
    batch_empty_text_embed = self.empty_text_embed.repeat((rgb_latent.shape
        [0], 1, 1))
    if show_pbar:
        iterable = tqdm(enumerate(timesteps), total=len(timesteps), leave=
            False, desc=' ' * 4 + 'Diffusion denoising')
    else:
        iterable = enumerate(timesteps)
    for i, t in iterable:
        unet_input = torch.cat([rgb_latent, depth_latent], dim=1)
        noise_pred = self.unet(unet_input, t, encoder_hidden_states=
            batch_empty_text_embed).sample
        depth_latent = self.scheduler.step(noise_pred, t, depth_latent,
            generator=rand_num_generator).prev_sample
    depth = self.decode_depth(depth_latent)
    depth = torch.clip(depth, -1.0, 1.0)
    depth = (depth + 1.0) / 2.0
    return depth
