def convert_model_output(self, model_output: torch.Tensor, *args, sample:
    torch.Tensor=None, **kwargs) ->torch.Tensor:
    """
        Convert the model output to the corresponding type the UniPC algorithm needs.

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
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
    sigma = self.sigmas[self.step_index]
    alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
    if self.predict_x0:
        if self.config.prediction_type == 'epsilon':
            x0_pred = (sample - sigma_t * model_output) / alpha_t
        elif self.config.prediction_type == 'sample':
            x0_pred = model_output
        elif self.config.prediction_type == 'v_prediction':
            x0_pred = alpha_t * sample - sigma_t * model_output
        else:
            raise ValueError(
                f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction` for the UniPCMultistepScheduler.'
                )
        if self.config.thresholding:
            x0_pred = self._threshold_sample(x0_pred)
        return x0_pred
    elif self.config.prediction_type == 'epsilon':
        return model_output
    elif self.config.prediction_type == 'sample':
        epsilon = (sample - alpha_t * model_output) / sigma_t
        return epsilon
    elif self.config.prediction_type == 'v_prediction':
        epsilon = alpha_t * model_output + sigma_t * sample
        return epsilon
    else:
        raise ValueError(
            f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction` for the UniPCMultistepScheduler.'
            )
