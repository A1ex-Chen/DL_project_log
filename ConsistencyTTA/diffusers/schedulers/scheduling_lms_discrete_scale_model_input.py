def scale_model_input(self, sample: torch.FloatTensor, timestep: Union[
    float, torch.FloatTensor]) ->torch.FloatTensor:
    """
        Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the K-LMS algorithm.

        Args:
            sample (`torch.FloatTensor`): input sample
            timestep (`float` or `torch.FloatTensor`): the current timestep in the diffusion chain

        Returns:
            `torch.FloatTensor`: scaled input sample
        """
    if isinstance(timestep, torch.Tensor):
        timestep = timestep.to(self.timesteps.device)
    step_index = (self.timesteps == timestep).nonzero().item()
    sigma = self.sigmas[step_index]
    sample = sample / (sigma ** 2 + 1) ** 0.5
    self.is_scale_input_called = True
    return sample
