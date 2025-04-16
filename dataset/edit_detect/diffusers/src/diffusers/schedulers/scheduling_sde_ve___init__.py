@register_to_config
def __init__(self, num_train_timesteps: int=2000, snr: float=0.15,
    sigma_min: float=0.01, sigma_max: float=1348.0, sampling_eps: float=
    1e-05, correct_steps: int=1):
    self.init_noise_sigma = sigma_max
    self.timesteps = None
    self.set_sigmas(num_train_timesteps, sigma_min, sigma_max, sampling_eps)
