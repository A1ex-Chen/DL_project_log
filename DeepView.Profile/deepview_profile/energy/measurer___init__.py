def __init__(self):
    self.sleep_interval = 0.1
    self.measuring = False
    self.measure_thread = None
    self.measurers = {'cpu': CPUMeasurer(self.sleep_interval), 'gpu':
        GPUMeasurer(self.sleep_interval)}
