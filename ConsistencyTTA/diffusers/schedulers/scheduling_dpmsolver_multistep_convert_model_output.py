def convert_model_output(self, model_output: torch.FloatTensor, timestep:
    int, sample: torch.FloatTensor) ->torch.FloatTensor:
    """
        Convert the model output to the corresponding type that the algorithm (DPM-Solver / DPM-Solver++) needs.

        DPM-Solver is designed to discretize an integral of the noise prediction model, and DPM-Solver++ is designed to
        discretize an integral of the data prediction model. So we need to first convert the model output to the
        corresponding type to match the algorithm.

        Note that the algorithm type and the model type is decoupled. That is to say, we can use either DPM-Solver or
        DPM-Solver++ for both noise prediction model and data prediction model.

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `torch.FloatTensor`: the converted model output.
        """
    if self.config.algorithm_type in ['dpmsolver++', 'sde-dpmsolver++']:
        if self.config.prediction_type == 'epsilon':
            if self.config.variance_type in ['learned', 'learned_range']:
                model_output = model_output[:, :3]
            alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
            x0_pred = (sample - sigma_t * model_output) / alpha_t
        elif self.config.prediction_type == 'sample':
            x0_pred = model_output
        elif self.config.prediction_type == 'v_prediction':
            alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
            x0_pred = alpha_t * sample - sigma_t * model_output
        else:
            raise ValueError(
                f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction` for the DPMSolverMultistepScheduler.'
                )
        if self.config.thresholding:
            x0_pred = self._threshold_sample(x0_pred)
        return x0_pred
    elif self.config.algorithm_type in ['dpmsolver', 'sde-dpmsolver']:
        if self.config.prediction_type == 'epsilon':
            if self.config.variance_type in ['learned', 'learned_range']:
                epsilon = model_output[:, :3]
            else:
                epsilon = model_output
        elif self.config.prediction_type == 'sample':
            alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
            epsilon = (sample - alpha_t * model_output) / sigma_t
        elif self.config.prediction_type == 'v_prediction':
            alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
            epsilon = alpha_t * model_output + sigma_t * sample
        else:
            raise ValueError(
                f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction` for the DPMSolverMultistepScheduler.'
                )
        if self.config.thresholding:
            alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
            x0_pred = (sample - sigma_t * epsilon) / alpha_t
            x0_pred = self._threshold_sample(x0_pred)
            epsilon = (sample - alpha_t * x0_pred) / sigma_t
        return epsilon
