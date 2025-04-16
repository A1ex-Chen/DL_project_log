def convert_model_output(self, model_output: torch.FloatTensor, timestep:
    int, sample: torch.FloatTensor) ->torch.FloatTensor:
    """
        Convert the model output to the corresponding type that the algorithm PC needs.

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `torch.FloatTensor`: the converted model output.
        """
    if self.predict_x0:
        if self.config.prediction_type == 'epsilon':
            alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
            x0_pred = (sample - sigma_t * model_output) / alpha_t
        elif self.config.prediction_type == 'sample':
            x0_pred = model_output
        elif self.config.prediction_type == 'v_prediction':
            alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
            x0_pred = alpha_t * sample - sigma_t * model_output
        else:
            raise ValueError(
                f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction` for the UniPCMultistepScheduler.'
                )
        if self.config.thresholding:
            orig_dtype = x0_pred.dtype
            if orig_dtype not in [torch.float, torch.double]:
                x0_pred = x0_pred.float()
            x0_pred = self._threshold_sample(x0_pred).type(orig_dtype)
        return x0_pred
    elif self.config.prediction_type == 'epsilon':
        return model_output
    elif self.config.prediction_type == 'sample':
        alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
        epsilon = (sample - alpha_t * model_output) / sigma_t
        return epsilon
    elif self.config.prediction_type == 'v_prediction':
        alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
        epsilon = alpha_t * model_output + sigma_t * sample
        return epsilon
    else:
        raise ValueError(
            f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction` for the UniPCMultistepScheduler.'
            )
