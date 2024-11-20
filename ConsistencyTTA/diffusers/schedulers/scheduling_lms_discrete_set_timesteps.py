def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.
    device]=None):
    """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, optional):
                the device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
    self.num_inference_steps = num_inference_steps
    timesteps = np.linspace(0, self.config.num_train_timesteps - 1,
        num_inference_steps, dtype=float)[::-1].copy()
    sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
    sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
    sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
    self.sigmas = torch.from_numpy(sigmas).to(device=device)
    if str(device).startswith('mps'):
        self.timesteps = torch.from_numpy(timesteps).to(device, dtype=torch
            .float32)
    else:
        self.timesteps = torch.from_numpy(timesteps).to(device=device)
    self.derivatives = []
