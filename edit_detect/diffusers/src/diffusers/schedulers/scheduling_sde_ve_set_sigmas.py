def set_sigmas(self, num_inference_steps: int, sigma_min: float=None,
    sigma_max: float=None, sampling_eps: float=None):
    """
        Sets the noise scales used for the diffusion chain (to be run before inference). The sigmas control the weight
        of the `drift` and `diffusion` components of the sample update.

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            sigma_min (`float`, optional):
                The initial noise scale value (overrides value given during scheduler instantiation).
            sigma_max (`float`, optional):
                The final noise scale value (overrides value given during scheduler instantiation).
            sampling_eps (`float`, optional):
                The final timestep value (overrides value given during scheduler instantiation).

        """
    sigma_min = sigma_min if sigma_min is not None else self.config.sigma_min
    sigma_max = sigma_max if sigma_max is not None else self.config.sigma_max
    sampling_eps = (sampling_eps if sampling_eps is not None else self.
        config.sampling_eps)
    if self.timesteps is None:
        self.set_timesteps(num_inference_steps, sampling_eps)
    self.sigmas = sigma_min * (sigma_max / sigma_min) ** (self.timesteps /
        sampling_eps)
    self.discrete_sigmas = torch.exp(torch.linspace(math.log(sigma_min),
        math.log(sigma_max), num_inference_steps))
    self.sigmas = torch.tensor([(sigma_min * (sigma_max / sigma_min) ** t) for
        t in self.timesteps])
