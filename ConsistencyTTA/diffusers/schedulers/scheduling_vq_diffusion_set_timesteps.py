def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.
    device]=None):
    """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.

            device (`str` or `torch.device`):
                device to place the timesteps and the diffusion process parameters (alpha, beta, gamma) on.
        """
    self.num_inference_steps = num_inference_steps
    timesteps = np.arange(0, self.num_inference_steps)[::-1].copy()
    self.timesteps = torch.from_numpy(timesteps).to(device)
    self.log_at = self.log_at.to(device)
    self.log_bt = self.log_bt.to(device)
    self.log_ct = self.log_ct.to(device)
    self.log_cumprod_at = self.log_cumprod_at.to(device)
    self.log_cumprod_bt = self.log_cumprod_bt.to(device)
    self.log_cumprod_ct = self.log_cumprod_ct.to(device)
