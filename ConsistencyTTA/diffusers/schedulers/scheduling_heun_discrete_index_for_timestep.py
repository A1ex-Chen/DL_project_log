def index_for_timestep(self, timestep):
    """Get the first / last index at which self.timesteps == timestep
        """
    assert len(timestep.shape) < 2
    avail_timesteps = self.timesteps.reshape(1, -1).to(timestep.device)
    mask = avail_timesteps == timestep.reshape(-1, 1)
    assert (mask.sum(dim=1) != 0).all(), f'timestep: {timestep.tolist()}'
    mask = mask.cpu() * torch.arange(mask.shape[1]).reshape(1, -1)
    if self.state_in_first_order:
        return mask.argmax(dim=1).numpy()
    else:
        return mask.argmax(dim=1).numpy() - 1
