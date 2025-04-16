def end_measurement(self):
    self.measuring = False
    self.measure_thread.join()
    self.measure_thread = None
