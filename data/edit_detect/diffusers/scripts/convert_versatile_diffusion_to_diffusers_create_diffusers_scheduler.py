def create_diffusers_scheduler(original_config):
    schedular = DDIMScheduler(num_train_timesteps=original_config.model.
        params.timesteps, beta_start=original_config.model.params.
        linear_start, beta_end=original_config.model.params.linear_end,
        beta_schedule='scaled_linear')
    return schedular
