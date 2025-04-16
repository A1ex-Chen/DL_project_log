def scale_model_input(self, sample: torch.Tensor, timestep: Union[float,
    torch.Tensor]) ->torch.Tensor:
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
    if self.step_index is None:
        self._init_step_index(timestep)
    if self.state_in_first_order:
        sigma = self.sigmas[self.step_index]
    else:
        sigma = self.sigmas_interpol[self.step_index]
    sample = sample / (sigma ** 2 + 1) ** 0.5
    return sample
