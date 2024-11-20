def add_noise_to_input(self, sample: torch.Tensor, sigma: float, generator:
    Optional[torch.Generator]=None) ->Tuple[torch.Tensor, float]:
    """
        Explicit Langevin-like "churn" step of adding noise to the sample according to a `gamma_i ≥ 0` to reach a
        higher noise level `sigma_hat = sigma_i + gamma_i*sigma_i`.

        Args:
            sample (`torch.Tensor`):
                The input sample.
            sigma (`float`):
            generator (`torch.Generator`, *optional*):
                A random number generator.
        """
    if self.config.s_min <= sigma <= self.config.s_max:
        gamma = min(self.config.s_churn / self.num_inference_steps, 2 ** 
            0.5 - 1)
    else:
        gamma = 0
    eps = self.config.s_noise * randn_tensor(sample.shape, generator=generator
        ).to(sample.device)
    sigma_hat = sigma + gamma * sigma
    sample_hat = sample + (sigma_hat ** 2 - sigma ** 2) ** 0.5 * eps
    return sample_hat, sigma_hat
