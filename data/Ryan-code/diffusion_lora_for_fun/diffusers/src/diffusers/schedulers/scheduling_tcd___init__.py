@register_to_config
def __init__(self, num_train_timesteps: int=1000, beta_start: float=0.00085,
    beta_end: float=0.012, beta_schedule: str='scaled_linear',
    trained_betas: Optional[Union[np.ndarray, List[float]]]=None,
    original_inference_steps: int=50, clip_sample: bool=False,
    clip_sample_range: float=1.0, set_alpha_to_one: bool=True, steps_offset:
    int=0, prediction_type: str='epsilon', thresholding: bool=False,
    dynamic_thresholding_ratio: float=0.995, sample_max_value: float=1.0,
    timestep_spacing: str='leading', timestep_scaling: float=10.0,
    rescale_betas_zero_snr: bool=False):
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
    self.final_alpha_cumprod = torch.tensor(1.0
        ) if set_alpha_to_one else self.alphas_cumprod[0]
    self.init_noise_sigma = 1.0
    self.num_inference_steps = None
    self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-
        1].copy().astype(np.int64))
    self.custom_timesteps = False
    self._step_index = None
    self._begin_index = None
