def stop(self):
    self._nvml.stop()
    self._executor.shutdown()
