def step(self, model_output: torch.Tensor, timestep: int, x_alpha: torch.Tensor
    ) ->torch.Tensor:
    """
        Predict the sample at the previous timestep by reversing the ODE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`): direct output from learned diffusion model. It is the direction from x0 to x1.
            timestep (`float`): current timestep in the diffusion chain.
            x_alpha (`torch.Tensor`): x_alpha sample for the current timestep

        Returns:
            `torch.Tensor`: the sample at the previous timestep

        """
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
    alpha = timestep / self.num_inference_steps
    alpha_next = (timestep + 1) / self.num_inference_steps
    d = model_output
    x_alpha = x_alpha + (alpha_next - alpha) * d
    return x_alpha
