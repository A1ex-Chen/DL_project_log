def next_step(self, model_output, timestep, sample):
    timestep, next_timestep = min(timestep - self.scheduler.config.
        num_train_timesteps // self.num_inference_steps, 999), timestep
    alpha_prod_t = self.scheduler.alphas_cumprod[timestep
        ] if timestep >= 0 else self.scheduler.final_alpha_cumprod
    alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output
        ) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = (alpha_prod_t_next ** 0.5 * next_original_sample +
        next_sample_direction)
    return next_sample
