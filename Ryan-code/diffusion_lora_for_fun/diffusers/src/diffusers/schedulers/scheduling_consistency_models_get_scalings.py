def get_scalings(self, sigma):
    sigma_data = self.config.sigma_data
    c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
    c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2) ** 0.5
    return c_skip, c_out
