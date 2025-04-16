def precondition_inputs(self, sample, sigma):
    c_in = 1 / (sigma ** 2 + self.config.sigma_data ** 2) ** 0.5
    scaled_sample = sample * c_in
    return scaled_sample
