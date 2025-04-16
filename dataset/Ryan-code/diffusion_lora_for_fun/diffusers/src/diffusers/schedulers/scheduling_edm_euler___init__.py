@register_to_config
def __init__(self, sigma_min: float=0.002, sigma_max: float=80.0,
    sigma_data: float=0.5, sigma_schedule: str='karras',
    num_train_timesteps: int=1000, prediction_type: str='epsilon', rho:
    float=7.0):
    if sigma_schedule not in ['karras', 'exponential']:
        raise ValueError(
            f'Wrong value for provided for `sigma_schedule={sigma_schedule!r}`.`'
            )
    self.num_inference_steps = None
    ramp = torch.linspace(0, 1, num_train_timesteps)
    if sigma_schedule == 'karras':
        sigmas = self._compute_karras_sigmas(ramp)
    elif sigma_schedule == 'exponential':
        sigmas = self._compute_exponential_sigmas(ramp)
    self.timesteps = self.precondition_noise(sigmas)
    self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
    self.is_scale_input_called = False
    self._step_index = None
    self._begin_index = None
    self.sigmas = self.sigmas.to('cpu')
