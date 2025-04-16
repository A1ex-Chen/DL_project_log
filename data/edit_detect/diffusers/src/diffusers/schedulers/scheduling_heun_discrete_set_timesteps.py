def set_timesteps(self, num_inference_steps: Optional[int]=None, device:
    Union[str, torch.device]=None, num_train_timesteps: Optional[int]=None,
    timesteps: Optional[List[int]]=None):
    """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            num_train_timesteps (`int`, *optional*):
                The number of diffusion steps used when training the model. If `None`, the default
                `num_train_timesteps` attribute is used.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, timesteps will be
                generated based on the `timestep_spacing` attribute. If `timesteps` is passed, `num_inference_steps`
                must be `None`, and `timestep_spacing` attribute will be ignored.
        """
    if num_inference_steps is None and timesteps is None:
        raise ValueError(
            'Must pass exactly one of `num_inference_steps` or `custom_timesteps`.'
            )
    if num_inference_steps is not None and timesteps is not None:
        raise ValueError(
            'Can only pass one of `num_inference_steps` or `custom_timesteps`.'
            )
    if timesteps is not None and self.config.use_karras_sigmas:
        raise ValueError(
            'Cannot use `timesteps` with `config.use_karras_sigmas = True`')
    num_inference_steps = num_inference_steps or len(timesteps)
    self.num_inference_steps = num_inference_steps
    num_train_timesteps = (num_train_timesteps or self.config.
        num_train_timesteps)
    if timesteps is not None:
        timesteps = np.array(timesteps, dtype=np.float32)
    elif self.config.timestep_spacing == 'linspace':
        timesteps = np.linspace(0, num_train_timesteps - 1,
            num_inference_steps, dtype=np.float32)[::-1].copy()
    elif self.config.timestep_spacing == 'leading':
        step_ratio = num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[:
            :-1].copy().astype(np.float32)
        timesteps += self.config.steps_offset
    elif self.config.timestep_spacing == 'trailing':
        step_ratio = num_train_timesteps / self.num_inference_steps
        timesteps = np.arange(num_train_timesteps, 0, -step_ratio).round(
            ).copy().astype(np.float32)
        timesteps -= 1
    else:
        raise ValueError(
            f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )
    sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
    log_sigmas = np.log(sigmas)
    sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
    if self.config.use_karras_sigmas:
        sigmas = self._convert_to_karras(in_sigmas=sigmas,
            num_inference_steps=self.num_inference_steps)
        timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in
            sigmas])
    sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
    sigmas = torch.from_numpy(sigmas).to(device=device)
    self.sigmas = torch.cat([sigmas[:1], sigmas[1:-1].repeat_interleave(2),
        sigmas[-1:]])
    timesteps = torch.from_numpy(timesteps)
    timesteps = torch.cat([timesteps[:1], timesteps[1:].repeat_interleave(2)])
    self.timesteps = timesteps.to(device=device)
    self.prev_derivative = None
    self.dt = None
    self._step_index = None
    self._begin_index = None
    self.sigmas = self.sigmas.to('cpu')
