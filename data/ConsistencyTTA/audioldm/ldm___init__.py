def __init__(self, device='cuda', first_stage_config=None,
    cond_stage_config=None, num_timesteps_cond=None, cond_stage_key='image',
    cond_stage_trainable=False, concat_mode=True, cond_stage_forward=None,
    conditioning_key=None, scale_factor=1.0, scale_by_std=False,
    base_learning_rate=None, *args, **kwargs):
    self.device = device
    self.learning_rate = base_learning_rate
    self.num_timesteps_cond = default(num_timesteps_cond, 1)
    self.scale_by_std = scale_by_std
    assert self.num_timesteps_cond <= kwargs['timesteps']
    if conditioning_key is None:
        conditioning_key = 'concat' if concat_mode else 'crossattn'
    if cond_stage_config == '__is_unconditional__':
        conditioning_key = None
    ckpt_path = kwargs.pop('ckpt_path', None)
    ignore_keys = kwargs.pop('ignore_keys', [])
    super().__init__(*args, conditioning_key=conditioning_key, **kwargs)
    self.concat_mode = concat_mode
    self.cond_stage_trainable = cond_stage_trainable
    self.cond_stage_key = cond_stage_key
    self.cond_stage_key_orig = cond_stage_key
    try:
        self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
    except:
        self.num_downs = 0
    if not scale_by_std:
        self.scale_factor = scale_factor
    else:
        self.register_buffer('scale_factor', torch.tensor(scale_factor))
    self.instantiate_first_stage(first_stage_config)
    self.instantiate_cond_stage(cond_stage_config)
    self.cond_stage_forward = cond_stage_forward
    self.clip_denoised = False
