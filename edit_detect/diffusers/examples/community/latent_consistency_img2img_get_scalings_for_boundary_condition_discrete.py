def get_scalings_for_boundary_condition_discrete(self, t):
    self.sigma_data = 0.5
    c_skip = self.sigma_data ** 2 / ((t / 0.1) ** 2 + self.sigma_data ** 2)
    c_out = t / 0.1 / ((t / 0.1) ** 2 + self.sigma_data ** 2) ** 0.5
    return c_skip, c_out
