def begin_measurement(self):
    assert self.measure_thread is None
    self.measure_thread = Thread(target=self.run_measure)
    self.measuring = True
    self.measure_thread.start()
