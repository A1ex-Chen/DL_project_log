def index_for_timestep(self, timestep, schedule_timesteps=None):
    if schedule_timesteps is None:
        schedule_timesteps = self.timesteps
    indices = (schedule_timesteps == timestep).nonzero()
    pos = 1 if len(indices) > 1 else 0
    return indices[pos].item()
