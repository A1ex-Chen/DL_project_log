def _initialize_measurement_noise_covariance(self, confidence: float=1
    ) ->np.ndarray:
    scale_factor = 1 / (confidence * self.confidence_scaler)
    return np.eye(4) * self.measurement_noise * scale_factor
