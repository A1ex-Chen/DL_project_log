@register_to_config
def __init__(self, num_train_timesteps: int=1000, beta_start: float=0.0001,
    beta_end: float=0.02, beta_schedule: str='linear', trained_betas:
    Optional[np.ndarray]=None, solver_order: int=2, prediction_type: str=
    'epsilon', thresholding: bool=False, dynamic_thresholding_ratio: float=
    0.995, sample_max_value: float=1.0, algorithm_type: str='dpmsolver++',
    solver_type: str='midpoint', lower_order_final: bool=True,
    use_karras_sigmas: Optional[bool]=False, lambda_min_clipped: float=-
    float('inf'), variance_type: Optional[str]=None):
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
    self.alpha_t = torch.sqrt(self.alphas_cumprod)
    self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
    self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)
    self.init_noise_sigma = 1.0
    if algorithm_type not in ['dpmsolver', 'dpmsolver++']:
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
    self.num_inference_steps = None
    timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps,
        dtype=np.float32)[::-1].copy()
    self.timesteps = torch.from_numpy(timesteps)
    self.model_outputs = [None] * solver_order
    self.sample = None
    self.order_list = self.get_order_list(num_train_timesteps)
    self.use_karras_sigmas = use_karras_sigmas
