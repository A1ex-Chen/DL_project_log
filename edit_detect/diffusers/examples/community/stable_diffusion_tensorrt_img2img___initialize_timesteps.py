def __initialize_timesteps(self, timesteps, strength):
    self.scheduler.set_timesteps(timesteps)
    offset = self.scheduler.steps_offset if hasattr(self.scheduler,
        'steps_offset') else 0
    init_timestep = int(timesteps * strength) + offset
    init_timestep = min(init_timestep, timesteps)
    t_start = max(timesteps - init_timestep + offset, 0)
    timesteps = self.scheduler.timesteps[t_start:].to(self.torch_device)
    return timesteps, t_start
