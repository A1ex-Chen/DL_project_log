def _measure_ms(self, runnable):
    for _ in range(self._warm_up):
        runnable()
    self._start_event.record()
    for _ in range(self._measure_for):
        runnable()
    self._end_event.record()
    torch.cuda.synchronize()
    return self._start_event.elapsed_time(self._end_event) / self._measure_for
