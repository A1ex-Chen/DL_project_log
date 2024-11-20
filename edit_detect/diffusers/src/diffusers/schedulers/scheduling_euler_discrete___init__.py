@register_to_config
def __init__(self, num_train_timesteps: int=1000, beta_start: float=0.0001,
    beta_end: float=0.02, beta_schedule: str='linear', trained_betas:
    Optional[Union[np.ndarray, List[float]]]=None, prediction_type: str=
    'epsilon', interpolation_type: str='linear', use_karras_sigmas:
    Optional[bool]=False, sigma_min: Optional[float]=None, sigma_max:
    Optional[float]=None, timestep_spacing: str='linspace', timestep_type:
    str='discrete', steps_offset: int=0, rescale_betas_zero_snr: bool=False,
    final_sigmas_type: str='zero'):
    if trained_betas is not None:
        self.betas = torch.tensor(trained_betas, dtype=torch.float32)
    elif beta_schedule == 'linear':
        self.betas = torch.linspace(beta_start, beta_end,
            num_train_timesteps, dtype=torch.float32)
    elif beta_schedule == 'scaled_linear':
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5,
            num_train_timesteps, dtype=torch.float32) ** 2
    elif beta_schedule == 'squaredcos_cap_v2':
        self.betas = betas_for_alpha_bar(num_train_timesteps)
    else:
        raise NotImplementedError(
            f'{beta_schedule} does is not implemented for {self.__class__}')
    if rescale_betas_zero_snr:
        self.betas = rescale_zero_terminal_snr(self.betas)
    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    if rescale_betas_zero_snr:
        self.alphas_cumprod[-1] = 2 ** -24
    sigmas = (((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5).flip(0)
    timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps,
        dtype=float)[::-1].copy()
    timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)
    self.num_inference_steps = None
    if timestep_type == 'continuous' and prediction_type == 'v_prediction':
        self.timesteps = torch.Tensor([(0.25 * sigma.log()) for sigma in
            sigmas])
    else:
        self.timesteps = timesteps
    self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
    self.is_scale_input_called = False
    self.use_karras_sigmas = use_karras_sigmas
    self._step_index = None
    self._begin_index = None
    self.sigmas = self.sigmas.to('cpu')
