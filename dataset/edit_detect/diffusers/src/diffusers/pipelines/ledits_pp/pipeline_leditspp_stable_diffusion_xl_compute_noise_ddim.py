def compute_noise_ddim(scheduler, prev_latents, latents, timestep,
    noise_pred, eta):
    prev_timestep = (timestep - scheduler.config.num_train_timesteps //
        scheduler.num_inference_steps)
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep
        ] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (latents - beta_prod_t ** 0.5 * noise_pred
        ) / alpha_prod_t ** 0.5
    if scheduler.config.clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
    variance = scheduler._get_variance(timestep, prev_timestep)
    std_dev_t = eta * variance ** 0.5
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2
        ) ** 0.5 * noise_pred
    mu_xt = (alpha_prod_t_prev ** 0.5 * pred_original_sample +
        pred_sample_direction)
    if variance > 0.0:
        noise = (prev_latents - mu_xt) / (variance ** 0.5 * eta)
    else:
        noise = torch.tensor([0.0]).to(latents.device)
    return noise, mu_xt + eta * variance ** 0.5 * noise
