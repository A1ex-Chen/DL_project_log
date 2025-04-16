def q_posterior(self, x_start, x_t, t):
    posterior_mean = extract_into_tensor(self.posterior_mean_coef1, t, x_t.
        shape) * x_start + extract_into_tensor(self.posterior_mean_coef2, t,
        x_t.shape) * x_t
    posterior_variance = extract_into_tensor(self.posterior_variance, t,
        x_t.shape)
    posterior_log_variance_clipped = extract_into_tensor(self.
        posterior_log_variance_clipped, t, x_t.shape)
    return posterior_mean, posterior_variance, posterior_log_variance_clipped
