@register_to_config
def __init__(self, num_train_timesteps: int=1000, beta_start: float=0.0001,
    beta_end: float=0.02, beta_schedule: str='linear', trained_betas:
    Optional[Union[np.ndarray, List[float]]]=None, solver_order: int=2,
    prediction_type: str='epsilon', thresholding: bool=False,
    dynamic_thresholding_ratio: float=0.995, sample_max_value: float=1.0,
    algorithm_type: str='dpmsolver++', solver_type: str='midpoint',
    lower_order_final: bool=True, euler_at_final: bool=False,
    use_karras_sigmas: Optional[bool]=False, use_lu_lambdas: Optional[bool]
    =False, final_sigmas_type: Optional[str]='zero', lambda_min_clipped:
    float=-float('inf'), variance_type: Optional[str]=None,
    timestep_spacing: str='linspace', steps_offset: int=0,
    rescale_betas_zero_snr: bool=False):
    if algorithm_type in ['dpmsolver', 'sde-dpmsolver']:
        deprecation_message = (
            f'algorithm_type {algorithm_type} is deprecated and will be removed in a future version. Choose from `dpmsolver++` or `sde-dpmsolver++` instead'
            )
        deprecate('algorithm_types dpmsolver and sde-dpmsolver', '1.0.0',
            deprecation_message)
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
    self.alpha_t = torch.sqrt(self.alphas_cumprod)
    self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
    self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)
    self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
    self.init_noise_sigma = 1.0
    if algorithm_type not in ['dpmsolver', 'dpmsolver++', 'sde-dpmsolver',
        'sde-dpmsolver++']:
        if algorithm_type == 'deis':
            self.register_to_config(algorithm_type='dpmsolver++')
        else:
            raise NotImplementedError(
                f'{algorithm_type} does is not implemented for {self.__class__}'
                )
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
    self.num_inference_steps = None
    timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps,
        dtype=np.float32)[::-1].copy()
    self.timesteps = torch.from_numpy(timesteps)
    self.model_outputs = [None] * solver_order
    self.lower_order_nums = 0
    self._step_index = None
    self._begin_index = None
    self.sigmas = self.sigmas.to('cpu')
