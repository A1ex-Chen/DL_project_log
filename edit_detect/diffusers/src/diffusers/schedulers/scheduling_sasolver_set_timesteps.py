def set_timesteps(self, num_inference_steps: int=None, device: Union[str,
    torch.device]=None):
    """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
    clipped_idx = torch.searchsorted(torch.flip(self.lambda_t, [0]), self.
        config.lambda_min_clipped)
    last_timestep = (self.config.num_train_timesteps - clipped_idx).numpy(
        ).item()
    if self.config.timestep_spacing == 'linspace':
        timesteps = np.linspace(0, last_timestep - 1, num_inference_steps + 1
            ).round()[::-1][:-1].copy().astype(np.int64)
    elif self.config.timestep_spacing == 'leading':
        step_ratio = last_timestep // (num_inference_steps + 1)
        timesteps = (np.arange(0, num_inference_steps + 1) * step_ratio).round(
            )[::-1][:-1].copy().astype(np.int64)
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
    if self.config.use_karras_sigmas:
        log_sigmas = np.log(sigmas)
        sigmas = np.flip(sigmas).copy()
        sigmas = self._convert_to_karras(in_sigmas=sigmas,
            num_inference_steps=num_inference_steps)
        timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in
            sigmas]).round()
        sigmas = np.concatenate([sigmas, sigmas[-1:]]).astype(np.float32)
    else:
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]
            ) ** 0.5
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)
    self.sigmas = torch.from_numpy(sigmas)
    self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=
        torch.int64)
    self.num_inference_steps = len(timesteps)
    self.model_outputs = [None] * max(self.config.predictor_order, self.
        config.corrector_order - 1)
    self.lower_order_nums = 0
    self.last_sample = None
    self._step_index = None
    self._begin_index = None
    self.sigmas = self.sigmas.to('cpu')
