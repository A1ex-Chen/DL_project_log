def set_timesteps(self, num_inference_steps: int, sampling_eps: float=None,
    device: Union[str, torch.device]=None):
    """
        Sets the continuous timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            sampling_eps (`float`, optional):
                final timestep value (overrides value given at Scheduler instantiation).

        """
    sampling_eps = (sampling_eps if sampling_eps is not None else self.
        config.sampling_eps)
    self.timesteps = torch.linspace(1, sampling_eps, num_inference_steps,
        device=device)
