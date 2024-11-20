def get_scalings_for_boundary_condition_discrete(self, timestep):
    self.sigma_data = 0.5
    scaled_timestep = timestep * self.config.timestep_scaling
    c_skip = self.sigma_data ** 2 / (scaled_timestep ** 2 + self.sigma_data **
        2)
    c_out = scaled_timestep / (scaled_timestep ** 2 + self.sigma_data ** 2
        ) ** 0.5
    return c_skip, c_out
