def get_adjacent_sigma(self, timesteps, t):
    return torch.where(timesteps == 0, torch.zeros_like(t.to(timesteps.
        device)), self.discrete_sigmas[timesteps - 1].to(timesteps.device))
