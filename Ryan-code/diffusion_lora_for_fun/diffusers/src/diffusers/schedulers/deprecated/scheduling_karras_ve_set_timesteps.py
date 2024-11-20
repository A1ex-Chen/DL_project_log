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
    timesteps = np.arange(0, self.num_inference_steps)[::-1].copy()
    self.timesteps = torch.from_numpy(timesteps).to(device)
    schedule = [(self.config.sigma_max ** 2 * (self.config.sigma_min ** 2 /
        self.config.sigma_max ** 2) ** (i / (num_inference_steps - 1))) for
        i in self.timesteps]
    self.schedule = torch.tensor(schedule, dtype=torch.float32, device=device)
