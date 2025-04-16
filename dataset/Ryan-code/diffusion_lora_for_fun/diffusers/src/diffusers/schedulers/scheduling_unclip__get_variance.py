def _get_variance(self, t, prev_timestep=None, predicted_variance=None,
    variance_type=None):
    if prev_timestep is None:
        prev_timestep = t - 1
    alpha_prod_t = self.alphas_cumprod[t]
    alpha_prod_t_prev = self.alphas_cumprod[prev_timestep
        ] if prev_timestep >= 0 else self.one
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    if prev_timestep == t - 1:
        beta = self.betas[t]
    else:
        beta = 1 - alpha_prod_t / alpha_prod_t_prev
    variance = beta_prod_t_prev / beta_prod_t * beta
    if variance_type is None:
        variance_type = self.config.variance_type
    if variance_type == 'fixed_small_log':
        variance = torch.log(torch.clamp(variance, min=1e-20))
        variance = torch.exp(0.5 * variance)
    elif variance_type == 'learned_range':
        min_log = variance.log()
        max_log = beta.log()
        frac = (predicted_variance + 1) / 2
        variance = frac * max_log + (1 - frac) * min_log
    return variance
