def set_timesteps(self, num_inference_steps, device: Union[str, torch.
    device]=None):
    self.timesteps = torch.linspace(1, self.config.sampling_eps,
        num_inference_steps, device=device)
