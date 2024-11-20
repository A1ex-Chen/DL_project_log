def index_for_timestep(self, timestep):
    indices = (self.timesteps == timestep).nonzero()
    if self.state_in_first_order:
        pos = -1
    else:
        pos = 0
    return indices[pos].item()
