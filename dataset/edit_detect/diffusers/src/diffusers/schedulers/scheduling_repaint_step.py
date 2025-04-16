def step(self, model_output: torch.Tensor, timestep: int, sample: torch.
    Tensor, original_image: torch.Tensor, mask: torch.Tensor, generator:
    Optional[torch.Generator]=None, return_dict: bool=True) ->Union[
    RePaintSchedulerOutput, Tuple]:
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
            original_image (`torch.Tensor`):
                The original image to inpaint on.
            mask (`torch.Tensor`):
                The mask where a value of 0.0 indicates which part of the original image to inpaint.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_repaint.RePaintSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_repaint.RePaintSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_repaint.RePaintSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.

        """
    t = timestep
    prev_timestep = (timestep - self.config.num_train_timesteps // self.
        num_inference_steps)
    alpha_prod_t = self.alphas_cumprod[t]
    alpha_prod_t_prev = self.alphas_cumprod[prev_timestep
        ] if prev_timestep >= 0 else self.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output
        ) / alpha_prod_t ** 0.5
    if self.config.clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
    device = model_output.device
    noise = randn_tensor(model_output.shape, generator=generator, device=
        device, dtype=model_output.dtype)
    std_dev_t = self.eta * self._get_variance(timestep) ** 0.5
    variance = 0
    if t > 0 and self.eta > 0:
        variance = std_dev_t * noise
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2
        ) ** 0.5 * model_output
    prev_unknown_part = (alpha_prod_t_prev ** 0.5 * pred_original_sample +
        pred_sample_direction + variance)
    prev_known_part = alpha_prod_t_prev ** 0.5 * original_image + (1 -
        alpha_prod_t_prev) ** 0.5 * noise
    pred_prev_sample = mask * prev_known_part + (1.0 - mask
        ) * prev_unknown_part
    if not return_dict:
        return pred_prev_sample, pred_original_sample
    return RePaintSchedulerOutput(prev_sample=pred_prev_sample,
        pred_original_sample=pred_original_sample)
