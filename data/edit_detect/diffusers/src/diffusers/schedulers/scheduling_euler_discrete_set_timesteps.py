def set_timesteps(self, num_inference_steps: int=None, device: Union[str,
    torch.device]=None, timesteps: Optional[List[int]]=None, sigmas:
    Optional[List[float]]=None):
    """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary timesteps schedule. If `None`, timesteps will be generated
                based on the `timestep_spacing` attribute. If `timesteps` is passed, `num_inference_steps` and `sigmas`
                must be `None`, and `timestep_spacing` attribute will be ignored.
            sigmas (`List[float]`, *optional*):
                Custom sigmas used to support arbitrary timesteps schedule schedule. If `None`, timesteps and sigmas
                will be generated based on the relevant scheduler attributes. If `sigmas` is passed,
                `num_inference_steps` and `timesteps` must be `None`, and the timesteps will be generated based on the
                custom sigmas schedule.
        """
    if timesteps is not None and sigmas is not None:
        raise ValueError('Only one of `timesteps` or `sigmas` should be set.')
    if num_inference_steps is None and timesteps is None and sigmas is None:
        raise ValueError(
            'Must pass exactly one of `num_inference_steps` or `timesteps` or `sigmas.'
            )
    if num_inference_steps is not None and (timesteps is not None or sigmas
         is not None):
        raise ValueError(
            'Can only pass one of `num_inference_steps` or `timesteps` or `sigmas`.'
            )
    if timesteps is not None and self.config.use_karras_sigmas:
        raise ValueError(
            'Cannot set `timesteps` with `config.use_karras_sigmas = True`.')
    if (timesteps is not None and self.config.timestep_type == 'continuous' and
        self.config.prediction_type == 'v_prediction'):
        raise ValueError(
            "Cannot set `timesteps` with `config.timestep_type = 'continuous'` and `config.prediction_type = 'v_prediction'`."
            )
    if num_inference_steps is None:
        num_inference_steps = len(timesteps) if timesteps is not None else len(
            sigmas) - 1
    self.num_inference_steps = num_inference_steps
    if sigmas is not None:
        log_sigmas = np.log(np.array(((1 - self.alphas_cumprod) / self.
            alphas_cumprod) ** 0.5))
        sigmas = np.array(sigmas).astype(np.float32)
        timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in
            sigmas[:-1]])
    else:
        if timesteps is not None:
            timesteps = np.array(timesteps).astype(np.float32)
        elif self.config.timestep_spacing == 'linspace':
            timesteps = np.linspace(0, self.config.num_train_timesteps - 1,
                num_inference_steps, dtype=np.float32)[::-1].copy()
        elif self.config.timestep_spacing == 'leading':
            step_ratio = (self.config.num_train_timesteps // self.
                num_inference_steps)
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round(
                )[::-1].copy().astype(np.float32)
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == 'trailing':
            step_ratio = (self.config.num_train_timesteps / self.
                num_inference_steps)
            timesteps = np.arange(self.config.num_train_timesteps, 0, -
                step_ratio).round().copy().astype(np.float32)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
                )
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) **
            0.5)
        log_sigmas = np.log(sigmas)
        if self.config.interpolation_type == 'linear':
            sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        elif self.config.interpolation_type == 'log_linear':
            sigmas = torch.linspace(np.log(sigmas[-1]), np.log(sigmas[0]), 
                num_inference_steps + 1).exp().numpy()
        else:
            raise ValueError(
                f"{self.config.interpolation_type} is not implemented. Please specify interpolation_type to either 'linear' or 'log_linear'"
                )
        if self.config.use_karras_sigmas:
            sigmas = self._convert_to_karras(in_sigmas=sigmas,
                num_inference_steps=self.num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for
                sigma in sigmas])
        if self.config.final_sigmas_type == 'sigma_min':
            sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]
                ) ** 0.5
        elif self.config.final_sigmas_type == 'zero':
            sigma_last = 0
        else:
            raise ValueError(
                f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}"
                )
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)
    sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
    if (self.config.timestep_type == 'continuous' and self.config.
        prediction_type == 'v_prediction'):
        self.timesteps = torch.Tensor([(0.25 * sigma.log()) for sigma in
            sigmas[:-1]]).to(device=device)
    else:
        self.timesteps = torch.from_numpy(timesteps.astype(np.float32)).to(
            device=device)
    self._step_index = None
    self._begin_index = None
    self.sigmas = sigmas.to('cpu')
