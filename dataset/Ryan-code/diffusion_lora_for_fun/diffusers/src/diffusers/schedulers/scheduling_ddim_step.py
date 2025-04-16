def step(self, model_output: torch.Tensor, timestep: int, sample: torch.
    Tensor, eta: float=0.0, use_clipped_model_output: bool=False, generator
    =None, variance_noise: Optional[torch.Tensor]=None, return_dict: bool=True
    ) ->Union[DDIMSchedulerOutput, Tuple]:
    """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            eta (`float`):
                The weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`, defaults to `False`):
                If `True`, computes "corrected" `model_output` from the clipped predicted original sample. Necessary
                because predicted original sample is clipped to [-1, 1] when `self.config.clip_sample` is `True`. If no
                clipping has happened, "corrected" `model_output` would coincide with the one provided as input and
                `use_clipped_model_output` has no effect.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            variance_noise (`torch.Tensor`):
                Alternative to generating noise with `generator` by directly providing the noise for the variance
                itself. Useful for methods such as [`CycleDiffusion`].
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddim.DDIMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
    prev_timestep = (timestep - self.config.num_train_timesteps // self.
        num_inference_steps)
    alpha_prod_t = self.alphas_cumprod[timestep]
    alpha_prod_t_prev = self.alphas_cumprod[prev_timestep
        ] if prev_timestep >= 0 else self.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    if self.config.prediction_type == 'epsilon':
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output
            ) / alpha_prod_t ** 0.5
        pred_epsilon = model_output
    elif self.config.prediction_type == 'sample':
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample
            ) / beta_prod_t ** 0.5
    elif self.config.prediction_type == 'v_prediction':
        pred_original_sample = (alpha_prod_t ** 0.5 * sample - beta_prod_t **
            0.5 * model_output)
        pred_epsilon = (alpha_prod_t ** 0.5 * model_output + beta_prod_t **
            0.5 * sample)
    else:
        raise ValueError(
            f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or `v_prediction`'
            )
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(-self.config.
            clip_sample_range, self.config.clip_sample_range)
    variance = self._get_variance(timestep, prev_timestep)
    std_dev_t = eta * variance ** 0.5
    if use_clipped_model_output:
        pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample
            ) / beta_prod_t ** 0.5
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2
        ) ** 0.5 * pred_epsilon
    prev_sample = (alpha_prod_t_prev ** 0.5 * pred_original_sample +
        pred_sample_direction)
    if eta > 0:
        if variance_noise is not None and generator is not None:
            raise ValueError(
                'Cannot pass both generator and variance_noise. Please make sure that either `generator` or `variance_noise` stays `None`.'
                )
        if variance_noise is None:
            variance_noise = randn_tensor(model_output.shape, generator=
                generator, device=model_output.device, dtype=model_output.dtype
                )
        variance = std_dev_t * variance_noise
        prev_sample = prev_sample + variance
    if not return_dict:
        return prev_sample,
    return DDIMSchedulerOutput(prev_sample=prev_sample,
        pred_original_sample=pred_original_sample)
