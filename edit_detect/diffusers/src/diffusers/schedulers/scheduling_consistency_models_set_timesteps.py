def set_timesteps(self, num_inference_steps: Optional[int]=None, device:
    Union[str, torch.device]=None, timesteps: Optional[List[int]]=None):
    """
        Sets the timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of equal spacing between timesteps is used. If `timesteps` is passed,
                `num_inference_steps` must be `None`.
        """
    if num_inference_steps is None and timesteps is None:
        raise ValueError(
            'Exactly one of `num_inference_steps` or `timesteps` must be supplied.'
            )
    if num_inference_steps is not None and timesteps is not None:
        raise ValueError(
            'Can only pass one of `num_inference_steps` or `timesteps`.')
    if timesteps is not None:
        for i in range(1, len(timesteps)):
            if timesteps[i] >= timesteps[i - 1]:
                raise ValueError('`timesteps` must be in descending order.')
        if timesteps[0] >= self.config.num_train_timesteps:
            raise ValueError(
                f'`timesteps` must start before `self.config.train_timesteps`: {self.config.num_train_timesteps}.'
                )
        timesteps = np.array(timesteps, dtype=np.int64)
        self.custom_timesteps = True
    else:
        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f'`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`: {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle maximal {self.config.num_train_timesteps} timesteps.'
                )
        self.num_inference_steps = num_inference_steps
        step_ratio = (self.config.num_train_timesteps // self.
            num_inference_steps)
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[:
            :-1].copy().astype(np.int64)
        self.custom_timesteps = False
    num_train_timesteps = self.config.num_train_timesteps
    ramp = timesteps[::-1].copy()
    ramp = ramp / (num_train_timesteps - 1)
    sigmas = self._convert_to_karras(ramp)
    timesteps = self.sigma_to_t(sigmas)
    sigmas = np.concatenate([sigmas, [self.config.sigma_min]]).astype(np.
        float32)
    self.sigmas = torch.from_numpy(sigmas).to(device=device)
    if str(device).startswith('mps'):
        self.timesteps = torch.from_numpy(timesteps).to(device, dtype=torch
            .float32)
    else:
        self.timesteps = torch.from_numpy(timesteps).to(device=device)
    self._step_index = None
    self._begin_index = None
    self.sigmas = self.sigmas.to('cpu')
