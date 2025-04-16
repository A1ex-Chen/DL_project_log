def _forward_sde(self, timestep, sample, guidance_scale, text_embeddings,
    steps, eta=1.0, lora_scale=None, generator=None):
    num_train_timesteps = len(self.scheduler)
    alphas_cumprod = self.scheduler.alphas_cumprod
    initial_alpha_cumprod = torch.tensor(1.0)
    prev_timestep = timestep + num_train_timesteps // steps
    alpha_prod_t = alphas_cumprod[timestep
        ] if timestep >= 0 else initial_alpha_cumprod
    alpha_prod_t_prev = alphas_cumprod[prev_timestep]
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    x_prev = (alpha_prod_t_prev / alpha_prod_t) ** 0.5 * sample + (1 - 
        alpha_prod_t_prev / alpha_prod_t) ** 0.5 * torch.randn(sample.size(
        ), dtype=sample.dtype, layout=sample.layout, device=self.device,
        generator=generator)
    eps = self._get_eps(x_prev, prev_timestep, guidance_scale,
        text_embeddings, lora_scale)
    sigma_t_prev = eta * (1 - alpha_prod_t) ** 0.5 * (1 - alpha_prod_t_prev /
        (1 - alpha_prod_t_prev) * (1 - alpha_prod_t) / alpha_prod_t) ** 0.5
    pred_original_sample = (x_prev - beta_prod_t_prev ** 0.5 * eps
        ) / alpha_prod_t_prev ** 0.5
    pred_sample_direction_coeff = (1 - alpha_prod_t - sigma_t_prev ** 2) ** 0.5
    noise = (sample - alpha_prod_t ** 0.5 * pred_original_sample - 
        pred_sample_direction_coeff * eps) / sigma_t_prev
    return x_prev, noise
