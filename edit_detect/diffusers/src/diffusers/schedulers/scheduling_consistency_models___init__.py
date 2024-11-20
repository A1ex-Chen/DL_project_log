@register_to_config
def __init__(self, num_train_timesteps: int=40, sigma_min: float=0.002,
    sigma_max: float=80.0, sigma_data: float=0.5, s_noise: float=1.0, rho:
    float=7.0, clip_denoised: bool=True):
    self.init_noise_sigma = sigma_max
    ramp = np.linspace(0, 1, num_train_timesteps)
    sigmas = self._convert_to_karras(ramp)
    timesteps = self.sigma_to_t(sigmas)
    self.num_inference_steps = None
    self.sigmas = torch.from_numpy(sigmas)
    self.timesteps = torch.from_numpy(timesteps)
    self.custom_timesteps = False
    self.is_scale_input_called = False
    self._step_index = None
    self._begin_index = None
    self.sigmas = self.sigmas.to('cpu')
