def convert_model_output(self, state: DPMSolverMultistepSchedulerState,
    model_output: jnp.ndarray, timestep: int, sample: jnp.ndarray
    ) ->jnp.ndarray:
    """
        Convert the model output to the corresponding type that the algorithm (DPM-Solver / DPM-Solver++) needs.

        DPM-Solver is designed to discretize an integral of the noise prediction model, and DPM-Solver++ is designed to
        discretize an integral of the data prediction model. So we need to first convert the model output to the
        corresponding type to match the algorithm.

        Note that the algorithm type and the model type is decoupled. That is to say, we can use either DPM-Solver or
        DPM-Solver++ for both noise prediction model and data prediction model.

        Args:
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.

        Returns:
            `jnp.ndarray`: the converted model output.
        """
    if self.config.algorithm_type == 'dpmsolver++':
        if self.config.prediction_type == 'epsilon':
            alpha_t, sigma_t = state.alpha_t[timestep], state.sigma_t[timestep]
            x0_pred = (sample - sigma_t * model_output) / alpha_t
        elif self.config.prediction_type == 'sample':
            x0_pred = model_output
        elif self.config.prediction_type == 'v_prediction':
            alpha_t, sigma_t = state.alpha_t[timestep], state.sigma_t[timestep]
            x0_pred = alpha_t * sample - sigma_t * model_output
        else:
            raise ValueError(
                f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`,  or `v_prediction` for the FlaxDPMSolverMultistepScheduler.'
                )
        if self.config.thresholding:
            dynamic_max_val = jnp.percentile(jnp.abs(x0_pred), self.config.
                dynamic_thresholding_ratio, axis=tuple(range(1, x0_pred.ndim)))
            dynamic_max_val = jnp.maximum(dynamic_max_val, self.config.
                sample_max_value * jnp.ones_like(dynamic_max_val))
            x0_pred = jnp.clip(x0_pred, -dynamic_max_val, dynamic_max_val
                ) / dynamic_max_val
        return x0_pred
    elif self.config.algorithm_type == 'dpmsolver':
        if self.config.prediction_type == 'epsilon':
            return model_output
        elif self.config.prediction_type == 'sample':
            alpha_t, sigma_t = state.alpha_t[timestep], state.sigma_t[timestep]
            epsilon = (sample - alpha_t * model_output) / sigma_t
            return epsilon
        elif self.config.prediction_type == 'v_prediction':
            alpha_t, sigma_t = state.alpha_t[timestep], state.sigma_t[timestep]
            epsilon = alpha_t * model_output + sigma_t * sample
            return epsilon
        else:
            raise ValueError(
                f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`,  or `v_prediction` for the FlaxDPMSolverMultistepScheduler.'
                )
