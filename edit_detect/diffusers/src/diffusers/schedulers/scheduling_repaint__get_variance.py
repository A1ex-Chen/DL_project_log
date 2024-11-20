def _get_variance(self, t):
    prev_timestep = (t - self.config.num_train_timesteps // self.
        num_inference_steps)
    alpha_prod_t = self.alphas_cumprod[t]
    alpha_prod_t_prev = self.alphas_cumprod[prev_timestep
        ] if prev_timestep >= 0 else self.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = beta_prod_t_prev / beta_prod_t * (1 - alpha_prod_t /
        alpha_prod_t_prev)
    return variance