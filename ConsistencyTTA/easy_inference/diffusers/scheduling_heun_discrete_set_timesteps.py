def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.
    device]=None, num_train_timesteps: Optional[int]=None):
    """
        Sets the timesteps used for the diffusion chain. 
        Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples
                with a pre-trained model.
            device (`str` or `torch.device`, optional):
                the device to which the timesteps should be moved to.
                If `None`, the timesteps are not moved.
        """
    self.num_inference_steps = num_inference_steps
    num_train_timesteps = (num_train_timesteps or self.config.
        num_train_timesteps)
    timesteps = np.linspace(0, num_train_timesteps - 1, num_inference_steps,
        dtype=float)[::-1].copy()
    sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
    log_sigmas = np.log(sigmas)
    sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
    if self.use_karras_sigmas:
        sigmas = self._convert_to_karras(in_sigmas=sigmas,
            num_inference_steps=self.num_inference_steps)
        timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in
            sigmas])
    sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
    sigmas = torch.from_numpy(sigmas).to(device=device)
    self.sigmas = torch.cat([sigmas[:1], sigmas[1:-1].repeat_interleave(2),
        sigmas[-1:]])
    self.init_noise_sigma = self.sigmas.max()
    timesteps = torch.from_numpy(timesteps)
    timesteps = torch.cat([timesteps[:1], timesteps[1:].repeat_interleave(2)])
    if 'mps' in str(device):
        timesteps = timesteps.float()
    self.timesteps = timesteps.to(device)
    self.prev_derivative = None
    self.dt = None
