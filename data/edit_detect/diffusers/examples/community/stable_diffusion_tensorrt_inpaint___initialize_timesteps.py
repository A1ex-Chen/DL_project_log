def __initialize_timesteps(self, num_inference_steps, strength):
    self.scheduler.set_timesteps(num_inference_steps)
    offset = self.scheduler.config.steps_offset if hasattr(self.scheduler,
        'steps_offset') else 0
    init_timestep = int(num_inference_steps * strength) + offset
    init_timestep = min(init_timestep, num_inference_steps)
    t_start = max(num_inference_steps - init_timestep + offset, 0)
    timesteps = self.scheduler.timesteps[t_start * self.scheduler.order:].to(
        self.torch_device)
    return timesteps, num_inference_steps - t_start
