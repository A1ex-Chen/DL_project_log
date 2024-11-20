def _get_variance(self, t, predicted_variance=None, variance_type=None):
    num_inference_steps = (self.num_inference_steps if self.
        num_inference_steps else self.config.num_train_timesteps)
    prev_t = t - self.config.num_train_timesteps // num_inference_steps
    alpha_prod_t = self.alphas_cumprod[t]
    alpha_prod_t_prev = torch.where(prev_t >= 0, self.alphas_cumprod[prev_t
        ], self.one)
    current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
    variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
    if variance_type is None:
        variance_type = self.config.variance_type
    if variance_type == 'fixed_small':
        variance = torch.clamp(variance, min=1e-20)
    elif variance_type == 'fixed_small_log':
        variance = torch.log(torch.clamp(variance, min=1e-20))
        variance = torch.exp(0.5 * variance)
    elif variance_type == 'fixed_large':
        variance = current_beta_t
    elif variance_type == 'fixed_large_log':
        variance = torch.log(current_beta_t)
    elif variance_type == 'learned':
        return predicted_variance
    elif variance_type == 'learned_range':
        min_log = torch.log(variance)
        max_log = torch.log(self.betas[t])
        frac = (predicted_variance + 1) / 2
        variance = frac * max_log + (1 - frac) * min_log
    return variance
