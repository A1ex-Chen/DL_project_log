def _batch_get_variance(self, t, prev_t):
    alpha_prod_t = self.alphas_cumprod[t]
    alpha_prod_t_prev = self.alphas_cumprod[torch.clip(prev_t, min=0)]
    alpha_prod_t_prev[prev_t < 0] = torch.tensor(1.0)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = beta_prod_t_prev / beta_prod_t * (1 - alpha_prod_t /
        alpha_prod_t_prev)
    return variance
