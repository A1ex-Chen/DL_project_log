@register_to_config
def __init__(self, num_train_timesteps: int=1000, beta_start: float=0.0001,
    beta_end: float=0.02, beta_schedule: str='linear', trained_betas:
    Optional[Union[np.ndarray, List[float]]]=None, clip_sample: bool=True,
    set_alpha_to_one: bool=True, steps_offset: int=0, prediction_type: str=
    'epsilon', clip_sample_range: float=1.0, timestep_spacing: str=
    'leading', rescale_betas_zero_snr: bool=False, **kwargs):
    if kwargs.get('set_alpha_to_zero', None) is not None:
        deprecation_message = (
            'The `set_alpha_to_zero` argument is deprecated. Please use `set_alpha_to_one` instead.'
            )
        deprecate('set_alpha_to_zero', '1.0.0', deprecation_message,
            standard_warn=False)
        set_alpha_to_one = kwargs['set_alpha_to_zero']
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
    self.initial_alpha_cumprod = torch.tensor(1.0
        ) if set_alpha_to_one else self.alphas_cumprod[0]
    self.init_noise_sigma = 1.0
    self.num_inference_steps = None
    self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps).
        copy().astype(np.int64))
