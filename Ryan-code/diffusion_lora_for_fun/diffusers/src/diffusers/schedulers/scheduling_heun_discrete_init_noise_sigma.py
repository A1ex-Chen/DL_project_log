@property
def init_noise_sigma(self):
    if self.config.timestep_spacing in ['linspace', 'trailing']:
        return self.sigmas.max()
    return (self.sigmas.max() ** 2 + 1) ** 0.5
