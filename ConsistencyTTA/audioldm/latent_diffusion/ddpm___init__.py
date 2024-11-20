def __init__(self, unet_config, timesteps=1000, beta_schedule='linear',
    loss_type='l2', ckpt_path=None, ignore_keys=[], load_only_unet=False,
    monitor='val/loss', use_ema=True, first_stage_key='image',
    latent_t_size=256, latent_f_size=16, channels=3, log_every_t=100,
    clip_denoised=True, linear_start=0.0001, linear_end=0.02, cosine_s=
    0.008, given_betas=None, original_elbo_weight=0.0, v_posterior=0.0,
    l_simple_weight=1.0, conditioning_key=None, parameterization='eps',
    scheduler_config=None, use_positional_encodings=False, learn_logvar=
    False, logvar_init=0.0):
    super().__init__()
    assert parameterization in ['eps', 'x0'
        ], 'currently only supporting "eps" and "x0"'
    self.parameterization = parameterization
    self.state = None
    self.cond_stage_model = None
    self.clip_denoised = clip_denoised
    self.log_every_t = log_every_t
    self.first_stage_key = first_stage_key
    self.latent_t_size = latent_t_size
    self.latent_f_size = latent_f_size
    self.channels = channels
    self.use_positional_encodings = use_positional_encodings
    self.model = DiffusionWrapper(unet_config, conditioning_key)
    count_params(self.model, verbose=True)
    self.use_ema = use_ema
    if self.use_ema:
        self.model_ema = LitEma(self.model)
    self.use_scheduler = scheduler_config is not None
    if self.use_scheduler:
        self.scheduler_config = scheduler_config
    self.v_posterior = v_posterior
    self.original_elbo_weight = original_elbo_weight
    self.l_simple_weight = l_simple_weight
    if monitor is not None:
        self.monitor = monitor
    self.register_schedule(given_betas=given_betas, beta_schedule=
        beta_schedule, timesteps=timesteps, linear_start=linear_start,
        linear_end=linear_end, cosine_s=cosine_s)
    self.loss_type = loss_type
    self.learn_logvar = learn_logvar
    self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,)
        )
    if self.learn_logvar:
        self.logvar = nn.Parameter(self.logvar, requires_grad=True)
    else:
        self.logvar = nn.Parameter(self.logvar, requires_grad=False)
    self.logger_save_dir = None
    self.logger_project = None
    self.logger_version = None
    self.label_indices_total = None
    self.metrics_buffer = {'val/kullback_leibler_divergence_sigmoid': 15.0,
        'val/kullback_leibler_divergence_softmax': 10.0, 'val/psnr': 0.0,
        'val/ssim': 0.0, 'val/inception_score_mean': 1.0,
        'val/inception_score_std': 0.0,
        'val/kernel_inception_distance_mean': 0.0,
        'val/kernel_inception_distance_std': 0.0,
        'val/frechet_inception_distance': 133.0,
        'val/frechet_audio_distance': 32.0}
    self.initial_learning_rate = None
