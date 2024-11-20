@torch.no_grad()
def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0,
    unconditional_conditioning=None, use_original_steps=False):
    timesteps = np.arange(self.ddpm_num_timesteps
        ) if use_original_steps else self.ddim_timesteps
    timesteps = timesteps[:t_start]
    time_range = np.flip(timesteps)
    total_steps = timesteps.shape[0]
    iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
    x_dec = x_latent
    for i, step in enumerate(iterator):
        index = total_steps - i - 1
        ts = torch.full((x_latent.shape[0],), step, device=x_latent.device,
            dtype=torch.long)
        x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index,
            use_original_steps=use_original_steps,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning)
    return x_dec
