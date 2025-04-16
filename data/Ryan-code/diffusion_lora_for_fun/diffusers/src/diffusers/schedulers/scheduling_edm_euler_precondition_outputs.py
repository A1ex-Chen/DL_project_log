def precondition_outputs(self, sample, model_output, sigma):
    sigma_data = self.config.sigma_data
    c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
    if self.config.prediction_type == 'epsilon':
        c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2) ** 0.5
    elif self.config.prediction_type == 'v_prediction':
        c_out = -sigma * sigma_data / (sigma ** 2 + sigma_data ** 2) ** 0.5
    else:
        raise ValueError(
            f'Prediction type {self.config.prediction_type} is not supported.')
    denoised = c_skip * sample + c_out * model_output
    return denoised
