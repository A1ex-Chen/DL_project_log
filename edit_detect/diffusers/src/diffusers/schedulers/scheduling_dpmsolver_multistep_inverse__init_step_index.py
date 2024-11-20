def _init_step_index(self, timestep):
    if isinstance(timestep, torch.Tensor):
        timestep = timestep.to(self.timesteps.device)
    index_candidates = (self.timesteps == timestep).nonzero()
    if len(index_candidates) == 0:
        step_index = len(self.timesteps) - 1
    elif len(index_candidates) > 1:
        step_index = index_candidates[1].item()
    else:
        step_index = index_candidates[0].item()
    self._step_index = step_index
