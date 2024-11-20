def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
    step_ratio = timesteps // ddim_timesteps
    self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio
        ).round().astype(np.int64) - 1
    self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
    self.ddim_alpha_cumprods_prev = np.asarray([alpha_cumprods[0]] +
        alpha_cumprods[self.ddim_timesteps[:-1]].tolist())
    self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
    self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
    self.ddim_alpha_cumprods_prev = torch.from_numpy(self.
        ddim_alpha_cumprods_prev)
