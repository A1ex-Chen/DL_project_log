@property
def init_noise_sigma(self):
    max_sigma = max(self.sigmas) if isinstance(self.sigmas, list
        ) else self.sigmas.max()
    if self.config.timestep_spacing in ['linspace', 'trailing']:
        return max_sigma
    return (max_sigma ** 2 + 1) ** 0.5
