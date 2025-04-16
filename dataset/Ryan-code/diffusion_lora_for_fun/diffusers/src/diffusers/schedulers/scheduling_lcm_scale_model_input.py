def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int]=None
    ) ->torch.Tensor:
    """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.Tensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.
        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
    return sample
