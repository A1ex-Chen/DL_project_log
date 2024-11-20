@register_to_config
def __init__(self, sigma_min: float=0.002, sigma_max: float=80.0,
    sigma_data: float=0.5, sigma_schedule: str='karras',
    num_train_timesteps: int=1000, prediction_type: str='epsilon', rho:
    float=7.0, solver_order: int=2, thresholding: bool=False,
    dynamic_thresholding_ratio: float=0.995, sample_max_value: float=1.0,
    algorithm_type: str='dpmsolver++', solver_type: str='midpoint',
    lower_order_final: bool=True, euler_at_final: bool=False,
    final_sigmas_type: Optional[str]='zero'):
    if algorithm_type not in ['dpmsolver++', 'sde-dpmsolver++']:
        if algorithm_type == 'deis':
            self.register_to_config(algorithm_type='dpmsolver++')
        else:
            raise NotImplementedError(
                f'{algorithm_type} is not implemented for {self.__class__}')
    if solver_type not in ['midpoint', 'heun']:
        if solver_type in ['logrho', 'bh1', 'bh2']:
            self.register_to_config(solver_type='midpoint')
        else:
            raise NotImplementedError(
                f'{solver_type} does is not implemented for {self.__class__}')
    if algorithm_type not in ['dpmsolver++', 'sde-dpmsolver++'
        ] and final_sigmas_type == 'zero':
        raise ValueError(
            f'`final_sigmas_type` {final_sigmas_type} is not supported for `algorithm_type` {algorithm_type}. Please choose `sigma_min` instead.'
            )
    ramp = torch.linspace(0, 1, num_train_timesteps)
    if sigma_schedule == 'karras':
        sigmas = self._compute_karras_sigmas(ramp)
    elif sigma_schedule == 'exponential':
        sigmas = self._compute_exponential_sigmas(ramp)
    self.timesteps = self.precondition_noise(sigmas)
    self.sigmas = self.sigmas = torch.cat([sigmas, torch.zeros(1, device=
        sigmas.device)])
    self.num_inference_steps = None
    self.model_outputs = [None] * solver_order
    self.lower_order_nums = 0
    self._step_index = None
    self._begin_index = None
    self.sigmas = self.sigmas.to('cpu')
