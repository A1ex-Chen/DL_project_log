def prev_step(self, model_output, timestep, sample):
    prev_timestep = (timestep - self.scheduler.config.num_train_timesteps //
        self.scheduler.num_inference_steps)
    alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep
        ] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output
        ) / alpha_prod_t ** 0.5
    pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
    prev_sample = (alpha_prod_t_prev ** 0.5 * pred_original_sample +
        pred_sample_direction)
    return prev_sample
