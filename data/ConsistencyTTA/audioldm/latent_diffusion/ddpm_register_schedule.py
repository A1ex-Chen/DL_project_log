def register_schedule(self, given_betas=None, beta_schedule='linear',
    timesteps=1000, linear_start=0.0001, linear_end=0.02, cosine_s=0.008):
    if exists(given_betas):
        betas = given_betas
    else:
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=
            linear_start, linear_end=linear_end, cosine_s=cosine_s)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    timesteps, = betas.shape
    self.num_timesteps = int(timesteps)
    self.linear_start = linear_start
    self.linear_end = linear_end
    assert alphas_cumprod.shape[0
        ] == self.num_timesteps, 'alphas have to be defined for each timestep'
    to_torch = partial(torch.tensor, dtype=torch.float32)
    self.register_buffer('betas', to_torch(betas))
    self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
    self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
    self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(
        alphas_cumprod)))
    self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(
        1.0 - alphas_cumprod)))
    self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(
        1.0 - alphas_cumprod)))
    self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1.0 /
        alphas_cumprod)))
    self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(
        1.0 / alphas_cumprod - 1)))
    posterior_variance = (1 - self.v_posterior) * betas * (1.0 -
        alphas_cumprod_prev) / (1.0 - alphas_cumprod
        ) + self.v_posterior * betas
    self.register_buffer('posterior_variance', to_torch(posterior_variance))
    self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(
        np.maximum(posterior_variance, 1e-20))))
    self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(
        alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
    self.register_buffer('posterior_mean_coef2', to_torch((1.0 -
        alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)))
    if self.parameterization == 'eps':
        lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance *
            to_torch(alphas) * (1 - self.alphas_cumprod))
    elif self.parameterization == 'x0':
        lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2.0 *
            1 - torch.Tensor(alphas_cumprod))
    else:
        raise NotImplementedError('mu not supported')
    lvlb_weights[0] = lvlb_weights[1]
    self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
    assert not torch.isnan(self.lvlb_weights).all()
