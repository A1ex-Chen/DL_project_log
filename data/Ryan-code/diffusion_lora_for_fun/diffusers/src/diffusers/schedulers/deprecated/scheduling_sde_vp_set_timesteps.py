def set_timesteps(self, num_inference_steps, device: Union[str, torch.
    device]=None):
    """
        Sets the continuous timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
    self.timesteps = torch.linspace(1, self.config.sampling_eps,
        num_inference_steps, device=device)
