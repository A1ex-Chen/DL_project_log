def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int]=None
    ) ->torch.Tensor:
    """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.Tensor`): input sample
            timestep (`int`, optional): current timestep

        Returns:
            `torch.Tensor`: scaled input sample
        """
    return sample
