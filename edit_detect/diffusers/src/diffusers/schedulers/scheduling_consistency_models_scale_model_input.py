def scale_model_input(self, sample: torch.Tensor, timestep: Union[float,
    torch.Tensor]) ->torch.Tensor:
    """
        Scales the consistency model input by `(sigma**2 + sigma_data**2) ** 0.5`.

        Args:
            sample (`torch.Tensor`):
                The input sample.
            timestep (`float` or `torch.Tensor`):
                The current timestep in the diffusion chain.

        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
    if self.step_index is None:
        self._init_step_index(timestep)
    sigma = self.sigmas[self.step_index]
    sample = sample / (sigma ** 2 + self.config.sigma_data ** 2) ** 0.5
    self.is_scale_input_called = True
    return sample
