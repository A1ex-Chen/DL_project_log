def scale_model_input(self, sample: torch.Tensor, timestep: Union[float,
    torch.Tensor]) ->torch.Tensor:
    """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep. Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.

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
    sigma = self.sigmas[self.step_index]
    sample = self.precondition_inputs(sample, sigma)
    self.is_scale_input_called = True
    return sample
