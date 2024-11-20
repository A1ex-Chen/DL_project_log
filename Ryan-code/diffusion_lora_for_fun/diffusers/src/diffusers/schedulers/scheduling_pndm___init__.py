@register_to_config
def __init__(self, num_train_timesteps: int=1000, beta_start: float=0.0001,
    beta_end: float=0.02, beta_schedule: str='linear', trained_betas:
    Optional[Union[np.ndarray, List[float]]]=None, skip_prk_steps: bool=
    False, set_alpha_to_one: bool=False, prediction_type: str='epsilon',
    timestep_spacing: str='leading', steps_offset: int=0):
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
    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.final_alpha_cumprod = torch.tensor(1.0
        ) if set_alpha_to_one else self.alphas_cumprod[0]
    self.init_noise_sigma = 1.0
    self.pndm_order = 4
    self.cur_model_output = 0
    self.counter = 0
    self.cur_sample = None
    self.ets = []
    self.num_inference_steps = None
    self._timesteps = np.arange(0, num_train_timesteps)[::-1].copy()
    self.prk_timesteps = None
    self.plms_timesteps = None
    self.timesteps = None
