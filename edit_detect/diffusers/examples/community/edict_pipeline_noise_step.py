def noise_step(self, base: torch.Tensor, model_input: torch.Tensor,
    model_output: torch.Tensor, timestep: torch.Tensor):
    prev_timestep = (timestep - self.scheduler.config.num_train_timesteps /
        self.scheduler.num_inference_steps)
    alpha_prod_t, beta_prod_t = self._get_alpha_and_beta(timestep)
    alpha_prod_t_prev, beta_prod_t_prev = self._get_alpha_and_beta(
        prev_timestep)
    a_t = (alpha_prod_t_prev / alpha_prod_t) ** 0.5
    b_t = -a_t * beta_prod_t ** 0.5 + beta_prod_t_prev ** 0.5
    next_model_input = (base - b_t * model_output) / a_t
    return model_input, next_model_input.to(base.dtype)
