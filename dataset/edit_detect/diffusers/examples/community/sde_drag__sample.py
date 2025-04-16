def _sample(self, timestep, sample, guidance_scale, text_embeddings, steps,
    sde=False, noise=None, eta=1.0, lora_scale=None, generator=None):
    num_train_timesteps = len(self.scheduler)
    alphas_cumprod = self.scheduler.alphas_cumprod
    final_alpha_cumprod = torch.tensor(1.0)
    eps = self._get_eps(sample, timestep, guidance_scale, text_embeddings,
        lora_scale)
    prev_timestep = timestep - num_train_timesteps // steps
    alpha_prod_t = alphas_cumprod[timestep]
    alpha_prod_t_prev = alphas_cumprod[prev_timestep
        ] if prev_timestep >= 0 else final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    sigma_t = eta * ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) ** 0.5 * (
        1 - alpha_prod_t / alpha_prod_t_prev) ** 0.5 if sde else 0
    pred_original_sample = (sample - beta_prod_t ** 0.5 * eps
        ) / alpha_prod_t ** 0.5
    pred_sample_direction_coeff = (1 - alpha_prod_t_prev - sigma_t ** 2) ** 0.5
    noise = torch.randn(sample.size(), dtype=sample.dtype, layout=sample.
        layout, device=self.device, generator=generator
        ) if noise is None else noise
    latent = (alpha_prod_t_prev ** 0.5 * pred_original_sample + 
        pred_sample_direction_coeff * eps + sigma_t * noise)
    return latent
