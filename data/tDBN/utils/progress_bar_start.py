def start(self, total_size):
    self._start = True
    self._step_times = []
    self._finished_sizes = []
    self._time_elapsed = 0.0
    self._current_time = time.time()
    self._total_size = total_size
    self._progress = 0
