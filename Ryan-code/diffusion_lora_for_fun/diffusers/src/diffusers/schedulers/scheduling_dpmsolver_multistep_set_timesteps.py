def set_timesteps(self, num_inference_steps: int=None, device: Union[str,
    torch.device]=None, timesteps: Optional[List[int]]=None):
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
        """
    if num_inference_steps is None and timesteps is None:
        raise ValueError(
            'Must pass exactly one of `num_inference_steps` or `timesteps`.')
    if num_inference_steps is not None and timesteps is not None:
        raise ValueError(
            'Can only pass one of `num_inference_steps` or `custom_timesteps`.'
            )
    if timesteps is not None and self.config.use_karras_sigmas:
        raise ValueError(
            'Cannot use `timesteps` with `config.use_karras_sigmas = True`')
    if timesteps is not None and self.config.use_lu_lambdas:
        raise ValueError(
            'Cannot use `timesteps` with `config.use_lu_lambdas = True`')
    if timesteps is not None:
        timesteps = np.array(timesteps).astype(np.int64)
    else:
        clipped_idx = torch.searchsorted(torch.flip(self.lambda_t, [0]),
            self.config.lambda_min_clipped)
        last_timestep = (self.config.num_train_timesteps - clipped_idx).numpy(
            ).item()
        if self.config.timestep_spacing == 'linspace':
            timesteps = np.linspace(0, last_timestep - 1, 
                num_inference_steps + 1).round()[::-1][:-1].copy().astype(np
                .int64)
        elif self.config.timestep_spacing == 'leading':
            step_ratio = last_timestep // (num_inference_steps + 1)
            timesteps = (np.arange(0, num_inference_steps + 1) * step_ratio
                ).round()[::-1][:-1].copy().astype(np.int64)
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == 'trailing':
            step_ratio = self.config.num_train_timesteps / num_inference_steps
            timesteps = np.arange(last_timestep, 0, -step_ratio).round().copy(
                ).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
                )
    sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
    log_sigmas = np.log(sigmas)
    if self.config.use_karras_sigmas:
        sigmas = np.flip(sigmas).copy()
        sigmas = self._convert_to_karras(in_sigmas=sigmas,
            num_inference_steps=num_inference_steps)
        timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in
            sigmas]).round()
    elif self.config.use_lu_lambdas:
        lambdas = np.flip(log_sigmas.copy())
        lambdas = self._convert_to_lu(in_lambdas=lambdas,
            num_inference_steps=num_inference_steps)
        sigmas = np.exp(lambdas)
        timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in
            sigmas]).round()
    else:
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
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
    self.sigmas = torch.from_numpy(sigmas)
    self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=
        torch.int64)
    self.num_inference_steps = len(timesteps)
    self.model_outputs = [None] * self.config.solver_order
    self.lower_order_nums = 0
    self._step_index = None
    self._begin_index = None
    self.sigmas = self.sigmas.to('cpu')
