def previous_timestep(self, timestep):
    index = (self.timesteps - timestep[0]).abs().argmin().item()
    prev_t = self.timesteps[index + 1][None].expand(timestep.shape[0])
    return prev_t
