@register_to_config
def __init__(self, num_train_timesteps: int=1000, beta_start: float=0.00085,
    beta_end: float=0.012, beta_schedule: str='linear', trained_betas:
    Optional[Union[np.ndarray, List[float]]]=None, prediction_type: str=
    'epsilon', use_karras_sigmas: Optional[bool]=False):
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
    self.use_karras_sigmas = use_karras_sigmas
    self.set_timesteps(num_train_timesteps, None, num_train_timesteps)
