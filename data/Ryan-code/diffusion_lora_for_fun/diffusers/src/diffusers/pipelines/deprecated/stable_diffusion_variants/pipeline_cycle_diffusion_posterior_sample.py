def posterior_sample(scheduler, latents, timestep, clean_latents, generator,
    eta):
    prev_timestep = (timestep - scheduler.config.num_train_timesteps //
        scheduler.num_inference_steps)
    if prev_timestep <= 0:
        return clean_latents
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep
        ] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
    variance = scheduler._get_variance(timestep, prev_timestep)
    std_dev_t = eta * variance ** 0.5
    e_t = (latents - alpha_prod_t ** 0.5 * clean_latents) / (1 - alpha_prod_t
        ) ** 0.5
    dir_xt = (1.0 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * e_t
    noise = std_dev_t * randn_tensor(clean_latents.shape, dtype=
        clean_latents.dtype, device=clean_latents.device, generator=generator)
    prev_latents = alpha_prod_t_prev ** 0.5 * clean_latents + dir_xt + noise
    return prev_latents
