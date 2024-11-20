def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[
    int]=None) ->torch.FloatTensor:
    """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.FloatTensor`):   input sample
            timestep (`int`, optional):     current timestep

        Returns:
            `torch.FloatTensor`: scaled input sample
        """
    return sample
