def index_for_timestep(self, timestep, schedule_timesteps=None):
    if schedule_timesteps is None:
        schedule_timesteps = self.timesteps
    index_candidates = (schedule_timesteps == timestep).nonzero()
    if len(index_candidates) == 0:
        step_index = len(self.timesteps) - 1
    elif len(index_candidates) > 1:
        step_index = index_candidates[1].item()
    else:
        step_index = index_candidates[0].item()
    return step_index
