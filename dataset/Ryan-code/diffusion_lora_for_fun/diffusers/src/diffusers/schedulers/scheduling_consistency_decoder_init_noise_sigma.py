@property
def init_noise_sigma(self):
    return self.sqrt_one_minus_alphas_cumprod[self.timesteps[0]]
