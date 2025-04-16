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
    steps = torch.linspace(1, 0, num_inference_steps + 1)[:-1]
    steps = torch.cat([steps, torch.tensor([0.0])])
    if self.config.trained_betas is not None:
        self.betas = torch.tensor(self.config.trained_betas, dtype=torch.
            float32)
    else:
        self.betas = torch.sin(steps * math.pi / 2) ** 2
    self.alphas = (1.0 - self.betas ** 2) ** 0.5
    timesteps = (torch.atan2(self.betas, self.alphas) / math.pi * 2)[:-1]
    self.timesteps = timesteps.to(device)
    self.ets = []
    self._step_index = None
    self._begin_index = None
