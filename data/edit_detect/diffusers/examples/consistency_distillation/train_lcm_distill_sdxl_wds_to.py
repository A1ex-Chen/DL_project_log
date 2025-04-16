def to(self, device):
    self.ddim_timesteps = self.ddim_timesteps.to(device)
    self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
    self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
    return self
