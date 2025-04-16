def step(self, model_output: torch.Tensor, timestep: int, sample: torch.
    Tensor, eta: float=0.3, generator: Optional[torch.Generator]=None,
    return_dict: bool=True) ->Union[TCDSchedulerOutput, Tuple]:
    """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            eta (`float`):
                A stochastic parameter (referred to as `gamma` in the paper) used to control the stochasticity in every
                step. When eta = 0, it represents deterministic sampling, whereas eta = 1 indicates full stochastic
                sampling.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_tcd.TCDSchedulerOutput`] or `tuple`.
        Returns:
            [`~schedulers.scheduling_utils.TCDSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_tcd.TCDSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
    if self.step_index is None:
        self._init_step_index(timestep)
    assert 0 <= eta <= 1.0, 'gamma must be less than or equal to 1.0'
    prev_step_index = self.step_index + 1
    if prev_step_index < len(self.timesteps):
        prev_timestep = self.timesteps[prev_step_index]
    else:
        prev_timestep = torch.tensor(0)
    timestep_s = torch.floor((1 - eta) * prev_timestep).to(dtype=torch.long)
    alpha_prod_t = self.alphas_cumprod[timestep]
    beta_prod_t = 1 - alpha_prod_t
    alpha_prod_t_prev = self.alphas_cumprod[prev_timestep
        ] if prev_timestep >= 0 else self.final_alpha_cumprod
    alpha_prod_s = self.alphas_cumprod[timestep_s]
    beta_prod_s = 1 - alpha_prod_s
    if self.config.prediction_type == 'epsilon':
        pred_original_sample = (sample - beta_prod_t.sqrt() * model_output
            ) / alpha_prod_t.sqrt()
        pred_epsilon = model_output
        pred_noised_sample = alpha_prod_s.sqrt(
            ) * pred_original_sample + beta_prod_s.sqrt() * pred_epsilon
    elif self.config.prediction_type == 'sample':
        pred_original_sample = model_output
        pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample
            ) / beta_prod_t ** 0.5
        pred_noised_sample = alpha_prod_s.sqrt(
            ) * pred_original_sample + beta_prod_s.sqrt() * pred_epsilon
    elif self.config.prediction_type == 'v_prediction':
        pred_original_sample = (alpha_prod_t ** 0.5 * sample - beta_prod_t **
            0.5 * model_output)
        pred_epsilon = (alpha_prod_t ** 0.5 * model_output + beta_prod_t **
            0.5 * sample)
        pred_noised_sample = alpha_prod_s.sqrt(
            ) * pred_original_sample + beta_prod_s.sqrt() * pred_epsilon
    else:
        raise ValueError(
            f'prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or `v_prediction` for `TCDScheduler`.'
            )
    if eta > 0:
        if self.step_index != self.num_inference_steps - 1:
            noise = randn_tensor(model_output.shape, generator=generator,
                device=model_output.device, dtype=pred_noised_sample.dtype)
            prev_sample = (alpha_prod_t_prev / alpha_prod_s).sqrt(
                ) * pred_noised_sample + (1 - alpha_prod_t_prev / alpha_prod_s
                ).sqrt() * noise
        else:
            prev_sample = pred_noised_sample
    else:
        prev_sample = pred_noised_sample
    self._step_index += 1
    if not return_dict:
        return prev_sample, pred_noised_sample
    return TCDSchedulerOutput(prev_sample=prev_sample, pred_noised_sample=
        pred_noised_sample)
