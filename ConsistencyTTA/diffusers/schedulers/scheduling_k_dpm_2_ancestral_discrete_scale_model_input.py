def scale_model_input(self, sample: torch.FloatTensor, timestep: Union[
    float, torch.FloatTensor]) ->torch.FloatTensor:
    """
        Args:
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.
            sample (`torch.FloatTensor`): input sample timestep (`int`, optional): current timestep
        Returns:
            `torch.FloatTensor`: scaled input sample
        """
    step_index = self.index_for_timestep(timestep)
    if self.state_in_first_order:
        sigma = self.sigmas[step_index]
    else:
        sigma = self.sigmas_interpol[step_index - 1]
    sample = sample / (sigma ** 2 + 1) ** 0.5
    return sample
