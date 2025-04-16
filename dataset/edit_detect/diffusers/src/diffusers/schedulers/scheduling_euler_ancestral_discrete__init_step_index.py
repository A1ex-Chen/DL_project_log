def _init_step_index(self, timestep):
    if self.begin_index is None:
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        self._step_index = self.index_for_timestep(timestep)
    else:
        self._step_index = self._begin_index
