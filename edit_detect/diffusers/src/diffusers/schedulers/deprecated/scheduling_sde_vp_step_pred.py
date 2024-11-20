def step_pred(self, score, x, t, generator=None):
    """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            score ():
            x ():
            t ():
            generator (`torch.Generator`, *optional*):
                A random number generator.
        """
    if self.timesteps is None:
        raise ValueError(
            "`self.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )
    log_mean_coeff = -0.25 * t ** 2 * (self.config.beta_max - self.config.
        beta_min) - 0.5 * t * self.config.beta_min
    std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
    std = std.flatten()
    while len(std.shape) < len(score.shape):
        std = std.unsqueeze(-1)
    score = -score / std
    dt = -1.0 / len(self.timesteps)
    beta_t = self.config.beta_min + t * (self.config.beta_max - self.config
        .beta_min)
    beta_t = beta_t.flatten()
    while len(beta_t.shape) < len(x.shape):
        beta_t = beta_t.unsqueeze(-1)
    drift = -0.5 * beta_t * x
    diffusion = torch.sqrt(beta_t)
    drift = drift - diffusion ** 2 * score
    x_mean = x + drift * dt
    noise = randn_tensor(x.shape, layout=x.layout, generator=generator,
        device=x.device, dtype=x.dtype)
    x = x_mean + diffusion * math.sqrt(-dt) * noise
    return x, x_mean
