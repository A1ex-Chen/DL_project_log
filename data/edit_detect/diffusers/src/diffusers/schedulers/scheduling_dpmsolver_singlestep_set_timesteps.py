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
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of equal spacing between timesteps schedule is used. If `timesteps` is
                passed, `num_inference_steps` must be `None`.
        """
    if num_inference_steps is None and timesteps is None:
        raise ValueError(
            'Must pass exactly one of  `num_inference_steps` or `timesteps`.')
    if num_inference_steps is not None and timesteps is not None:
        raise ValueError(
            'Must pass exactly one of  `num_inference_steps` or `timesteps`.')
    if timesteps is not None and self.config.use_karras_sigmas:
        raise ValueError(
            'Cannot use `timesteps` when `config.use_karras_sigmas=True`.')
    num_inference_steps = num_inference_steps or len(timesteps)
    self.num_inference_steps = num_inference_steps
    if timesteps is not None:
        timesteps = np.array(timesteps).astype(np.int64)
    else:
        clipped_idx = torch.searchsorted(torch.flip(self.lambda_t, [0]),
            self.config.lambda_min_clipped)
        timesteps = np.linspace(0, self.config.num_train_timesteps - 1 -
            clipped_idx, num_inference_steps + 1).round()[::-1][:-1].copy(
            ).astype(np.int64)
    sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
    if self.config.use_karras_sigmas:
        log_sigmas = np.log(sigmas)
        sigmas = np.flip(sigmas).copy()
        sigmas = self._convert_to_karras(in_sigmas=sigmas,
            num_inference_steps=num_inference_steps)
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
            f' `final_sigmas_type` must be one of `sigma_min` or `zero`, but got {self.config.final_sigmas_type}'
            )
    sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)
    self.sigmas = torch.from_numpy(sigmas).to(device=device)
    self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=
        torch.int64)
    self.model_outputs = [None] * self.config.solver_order
    self.sample = None
    if (not self.config.lower_order_final and num_inference_steps % self.
        config.solver_order != 0):
        logger.warning(
            'Changing scheduler {self.config} to have `lower_order_final` set to True to handle uneven amount of inference steps. Please make sure to always use an even number of `num_inference steps when using `lower_order_final=False`.'
            )
        self.register_to_config(lower_order_final=True)
    if (not self.config.lower_order_final and self.config.final_sigmas_type ==
        'zero'):
        logger.warning(
            " `last_sigmas_type='zero'` is not supported for `lower_order_final=False`. Changing scheduler {self.config} to have `lower_order_final` set to True."
            )
        self.register_to_config(lower_order_final=True)
    self.order_list = self.get_order_list(num_inference_steps)
    self._step_index = None
    self._begin_index = None
    self.sigmas = self.sigmas.to('cpu')
