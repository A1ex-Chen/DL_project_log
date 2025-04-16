def p_mean_variance(self, x, t, clip_denoised: bool):
    model_out = self.model(x, t)
    if self.parameterization == 'eps':
        x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
    elif self.parameterization == 'x0':
        x_recon = model_out
    if clip_denoised:
        x_recon.clamp_(-1.0, 1.0)
    model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
        x_start=x_recon, x_t=x, t=t)
    return model_mean, posterior_variance, posterior_log_variance
