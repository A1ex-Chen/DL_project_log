def step(self, model_output: torch.FloatTensor, timestep: int, sample:
    torch.FloatTensor, original_image: torch.FloatTensor, mask: torch.
    FloatTensor, generator: Optional[torch.Generator]=None, return_dict:
    bool=True) ->Union[RePaintSchedulerOutput, Tuple]:
    """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned
                diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            original_image (`torch.FloatTensor`):
                the original image to inpaint on.
            mask (`torch.FloatTensor`):
                the mask where 0.0 values define which part of the original image to inpaint (change).
            generator (`torch.Generator`, *optional*): random number generator.
            return_dict (`bool`): option for returning tuple rather than
                DDPMSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.RePaintSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.RePaintSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

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
