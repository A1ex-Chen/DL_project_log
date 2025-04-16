@register_to_config
def __init__(self, num_train_timesteps: int=1000, beta_start: float=0.0001,
    beta_end: float=0.02, beta_schedule: str='linear', trained_betas:
    Optional[Union[np.ndarray, List[float]]]=None, solver_order: int=2,
    prediction_type: str='epsilon', thresholding: bool=False,
    dynamic_thresholding_ratio: float=0.995, sample_max_value: float=1.0,
    predict_x0: bool=True, solver_type: str='bh2', lower_order_final: bool=
    True, disable_corrector: List[int]=[], solver_p: SchedulerMixin=None):
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
    if solver_type not in ['bh1', 'bh2']:
        if solver_type in ['midpoint', 'heun', 'logrho']:
            self.register_to_config(solver_type='bh1')
        else:
            raise NotImplementedError(
                f'{solver_type} does is not implemented for {self.__class__}')
    self.predict_x0 = predict_x0
    self.num_inference_steps = None
    timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps,
        dtype=np.float32)[::-1].copy()
    self.timesteps = torch.from_numpy(timesteps)
    self.model_outputs = [None] * solver_order
    self.timestep_list = [None] * solver_order
    self.lower_order_nums = 0
    self.disable_corrector = disable_corrector
    self.solver_p = solver_p
    self.last_sample = None
