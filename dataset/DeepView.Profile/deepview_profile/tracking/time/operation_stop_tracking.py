def stop_tracking(self):
    super().stop_tracking()
    self._callable_tracker.stop_tracking()
