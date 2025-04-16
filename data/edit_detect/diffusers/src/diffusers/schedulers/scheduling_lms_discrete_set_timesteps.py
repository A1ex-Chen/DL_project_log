def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.
    device]=None):
    """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
    self.num_inference_steps = num_inference_steps
    if self.config.timestep_spacing == 'linspace':
        timesteps = np.linspace(0, self.config.num_train_timesteps - 1,
            num_inference_steps, dtype=np.float32)[::-1].copy()
    elif self.config.timestep_spacing == 'leading':
        step_ratio = (self.config.num_train_timesteps // self.
            num_inference_steps)
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[:
            :-1].copy().astype(np.float32)
        timesteps += self.config.steps_offset
    elif self.config.timestep_spacing == 'trailing':
        step_ratio = self.config.num_train_timesteps / self.num_inference_steps
        timesteps = np.arange(self.config.num_train_timesteps, 0, -step_ratio
            ).round().copy().astype(np.float32)
        timesteps -= 1
    else:
        raise ValueError(
            f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )
    sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
    log_sigmas = np.log(sigmas)
    sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
    if self.config.use_karras_sigmas:
        sigmas = self._convert_to_karras(in_sigmas=sigmas)
        timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in
            sigmas])
    sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
    self.sigmas = torch.from_numpy(sigmas).to(device=device)
    self.timesteps = torch.from_numpy(timesteps).to(device=device)
    self._step_index = None
    self._begin_index = None
    self.sigmas = self.sigmas.to('cpu')
    self.derivatives = []
