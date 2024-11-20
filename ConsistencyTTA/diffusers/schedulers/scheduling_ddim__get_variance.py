def _get_variance(self, timestep, prev_timestep):
    alpha_prod_t = self.alphas_cumprod[timestep]
    alpha_prod_t_prev = torch.where(prev_timestep >= 0, self.alphas_cumprod
        [prev_timestep], self.final_alpha_cumprod)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    return beta_prod_t_prev / beta_prod_t * (1 - alpha_prod_t /
        alpha_prod_t_prev)
