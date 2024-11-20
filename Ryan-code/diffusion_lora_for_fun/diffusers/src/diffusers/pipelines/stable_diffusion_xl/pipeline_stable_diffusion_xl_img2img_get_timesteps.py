def get_timesteps(self, num_inference_steps, strength, device,
    denoising_start=None):
    if denoising_start is None:
        init_timestep = min(int(num_inference_steps * strength),
            num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
    else:
        t_start = 0
    timesteps = self.scheduler.timesteps[t_start * self.scheduler.order:]
    if denoising_start is not None:
        discrete_timestep_cutoff = int(round(self.scheduler.config.
            num_train_timesteps - denoising_start * self.scheduler.config.
            num_train_timesteps))
        num_inference_steps = (timesteps < discrete_timestep_cutoff).sum(
            ).item()
        if self.scheduler.order == 2 and num_inference_steps % 2 == 0:
            num_inference_steps = num_inference_steps + 1
        timesteps = timesteps[-num_inference_steps:]
        return timesteps, num_inference_steps
    return timesteps, num_inference_steps - t_start
