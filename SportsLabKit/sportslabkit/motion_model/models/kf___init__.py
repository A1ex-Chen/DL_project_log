def __init__(self, dt: float=1 / 30, process_noise: float=0.001,
    measurement_noise: float=0.001, confidence_scaler: float=1.0):
    super().__init__()
    self.dt = dt
    self.process_noise = process_noise
    self.measurement_noise = measurement_noise
    self.confidence_scaler = confidence_scaler
