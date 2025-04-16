def convert_model_output(self, model_output: torch.FloatTensor, timestep:
    int, sample: torch.FloatTensor) ->torch.FloatTensor:
    """
        Convert the model output to the corresponding type that the algorithm DEIS needs.

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.

        Returns:
            `torch.FloatTensor`: the converted model output.
        """
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
            f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction` for the DEISMultistepScheduler.'
            )
    if self.config.thresholding:
        orig_dtype = x0_pred.dtype
        if orig_dtype not in [torch.float, torch.double]:
            x0_pred = x0_pred.float()
        x0_pred = self._threshold_sample(x0_pred).type(orig_dtype)
    if self.config.algorithm_type == 'deis':
        alpha_t, sigma_t = self.alpha_t[timestep], self.sigma_t[timestep]
        return (sample - alpha_t * x0_pred) / sigma_t
    else:
        raise NotImplementedError('only support log-rho multistep deis now')
