def scale_model_input(self, sample: torch.FloatTensor, timestep: Union[
    float, torch.FloatTensor]) ->torch.FloatTensor:
    """
        Ensures interchangeability with schedulers that need to scale the 
        denoising model input depending on the current timestep.
        Args:
            sample (`torch.FloatTensor`): input sample 
            timestep (`int`, optional): current timestep
        Returns:
            `torch.FloatTensor`: scaled input sample
        """
    if not torch.is_tensor(timestep):
        timestep = torch.tensor(timestep)
    timestep = timestep.to(sample.device).reshape(-1)
    step_index = self.index_for_timestep(timestep)
    sigma = self.sigmas[step_index].reshape(-1, 1, 1, 1).to(sample.device)
    sample = sample / (sigma ** 2 + 1) ** 0.5
    return sample
