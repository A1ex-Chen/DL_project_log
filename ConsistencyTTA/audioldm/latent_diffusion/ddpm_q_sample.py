def q_sample(self, x_start, t, noise=None):
    noise = default(noise, lambda : torch.randn_like(x_start))
    return extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape
        ) * x_start + extract_into_tensor(self.
        sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
