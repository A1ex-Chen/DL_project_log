def predict_start_from_noise(self, x_t, t, noise):
    return extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape
        ) * x_t - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t,
        x_t.shape) * noise
