def step(self, model_output: torch.FloatTensor, timestep: int, sample:
    torch.FloatTensor, eta: float=0.0, use_clipped_model_output: bool=False,
    variance_noise: Optional[torch.FloatTensor]=None, return_dict: bool=True
    ) ->Union[DDIMSchedulerOutput, Tuple]:
    prev_timestep = (timestep + self.config.num_train_timesteps // self.
        num_inference_steps)
    alpha_prod_t = self.alphas_cumprod[timestep]
    alpha_prod_t_prev = (self.alphas_cumprod[prev_timestep] if 
        prev_timestep < self.config.num_train_timesteps else self.
        final_alpha_cumprod)
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
    if self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(-self.config.
            clip_sample_range, self.config.clip_sample_range)
    pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * pred_epsilon
    prev_sample = (alpha_prod_t_prev ** 0.5 * pred_original_sample +
        pred_sample_direction)
    if not return_dict:
        return prev_sample, pred_original_sample
    return DDIMSchedulerOutput(prev_sample=prev_sample,
        pred_original_sample=pred_original_sample)
