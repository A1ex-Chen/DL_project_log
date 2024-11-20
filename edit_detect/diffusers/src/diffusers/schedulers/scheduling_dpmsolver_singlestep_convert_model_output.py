def convert_model_output(self, model_output: torch.Tensor, *args, sample:
    torch.Tensor=None, **kwargs) ->torch.Tensor:
    """
        Convert the model output to the corresponding type the DPMSolver/DPMSolver++ algorithm needs. DPM-Solver is
        designed to discretize an integral of the noise prediction model, and DPM-Solver++ is designed to discretize an
        integral of the data prediction model.

        <Tip>

        The algorithm and model type are decoupled. You can use either DPMSolver or DPMSolver++ for both noise
        prediction and data prediction models.

        </Tip>

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.Tensor`:
                The converted model output.
        """
    timestep = args[0] if len(args) > 0 else kwargs.pop('timestep', None)
    if sample is None:
        if len(args) > 1:
            sample = args[1]
        else:
            raise ValueError('missing `sample` as a required keyward argument')
    if timestep is not None:
        deprecate('timesteps', '1.0.0',
            'Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`'
            )
    if self.config.algorithm_type == 'dpmsolver++':
        if self.config.prediction_type == 'epsilon':
            if self.config.variance_type in ['learned_range']:
                model_output = model_output[:, :3]
            sigma = self.sigmas[self.step_index]
            alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
            x0_pred = (sample - sigma_t * model_output) / alpha_t
        elif self.config.prediction_type == 'sample':
            x0_pred = model_output
        elif self.config.prediction_type == 'v_prediction':
            sigma = self.sigmas[self.step_index]
            alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
            x0_pred = alpha_t * sample - sigma_t * model_output
        else:
            raise ValueError(
                f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction` for the DPMSolverSinglestepScheduler.'
                )
        if self.config.thresholding:
            x0_pred = self._threshold_sample(x0_pred)
        return x0_pred
    elif self.config.algorithm_type == 'dpmsolver':
        if self.config.prediction_type == 'epsilon':
            if self.config.variance_type in ['learned_range']:
                model_output = model_output[:, :3]
            return model_output
        elif self.config.prediction_type == 'sample':
            sigma = self.sigmas[self.step_index]
            alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
            epsilon = (sample - alpha_t * model_output) / sigma_t
            return epsilon
        elif self.config.prediction_type == 'v_prediction':
            sigma = self.sigmas[self.step_index]
            alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
            epsilon = alpha_t * model_output + sigma_t * sample
            return epsilon
        else:
            raise ValueError(
                f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction` for the DPMSolverSinglestepScheduler.'
                )
