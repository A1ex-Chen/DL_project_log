@property
def init_noise_sigma(self):
    return (self.config.sigma_max ** 2 + 1) ** 0.5
