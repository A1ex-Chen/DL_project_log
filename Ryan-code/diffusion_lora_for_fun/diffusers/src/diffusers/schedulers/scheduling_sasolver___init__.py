@register_to_config
def __init__(self, num_train_timesteps: int=1000, beta_start: float=0.0001,
    beta_end: float=0.02, beta_schedule: str='linear', trained_betas:
    Optional[Union[np.ndarray, List[float]]]=None, predictor_order: int=2,
    corrector_order: int=2, prediction_type: str='epsilon', tau_func:
    Optional[Callable]=None, thresholding: bool=False,
    dynamic_thresholding_ratio: float=0.995, sample_max_value: float=1.0,
    algorithm_type: str='data_prediction', lower_order_final: bool=True,
    use_karras_sigmas: Optional[bool]=False, lambda_min_clipped: float=-
    float('inf'), variance_type: Optional[str]=None, timestep_spacing: str=
    'linspace', steps_offset: int=0):
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
    self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
    self.init_noise_sigma = 1.0
    if algorithm_type not in ['data_prediction', 'noise_prediction']:
        raise NotImplementedError(
            f'{algorithm_type} does is not implemented for {self.__class__}')
    self.num_inference_steps = None
    timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps,
        dtype=np.float32)[::-1].copy()
    self.timesteps = torch.from_numpy(timesteps)
    self.timestep_list = [None] * max(predictor_order, corrector_order - 1)
    self.model_outputs = [None] * max(predictor_order, corrector_order - 1)
    if tau_func is None:
        self.tau_func = lambda t: 1 if t >= 200 and t <= 800 else 0
    else:
        self.tau_func = tau_func
    self.predict_x0 = algorithm_type == 'data_prediction'
    self.lower_order_nums = 0
    self.last_sample = None
    self._step_index = None
    self._begin_index = None
    self.sigmas = self.sigmas.to('cpu')
