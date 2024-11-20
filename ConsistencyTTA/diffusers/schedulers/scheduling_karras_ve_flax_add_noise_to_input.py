def add_noise_to_input(self, state: KarrasVeSchedulerState, sample: jnp.
    ndarray, sigma: float, key: random.KeyArray) ->Tuple[jnp.ndarray, float]:
    """
        Explicit Langevin-like "churn" step of adding noise to the sample according to a factor gamma_i ≥ 0 to reach a
        higher noise level sigma_hat = sigma_i + gamma_i*sigma_i.

        TODO Args:
        """
    if self.config.s_min <= sigma <= self.config.s_max:
        gamma = min(self.config.s_churn / state.num_inference_steps, 2 ** 
            0.5 - 1)
    else:
        gamma = 0
    key = random.split(key, num=1)
    eps = self.config.s_noise * random.normal(key=key, shape=sample.shape)
    sigma_hat = sigma + gamma * sigma
    sample_hat = sample + (sigma_hat ** 2 - sigma ** 2) ** 0.5 * eps
    return sample_hat, sigma_hat
