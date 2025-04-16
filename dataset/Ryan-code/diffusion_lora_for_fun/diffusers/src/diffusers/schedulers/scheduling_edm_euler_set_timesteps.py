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
    ramp = np.linspace(0, 1, self.num_inference_steps)
    if self.config.sigma_schedule == 'karras':
        sigmas = self._compute_karras_sigmas(ramp)
    elif self.config.sigma_schedule == 'exponential':
        sigmas = self._compute_exponential_sigmas(ramp)
    sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
    self.timesteps = self.precondition_noise(sigmas)
    self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
    self._step_index = None
    self._begin_index = None
    self.sigmas = self.sigmas.to('cpu')
