def _prepare_for_memory_profiling(self):
    if self._profiler is not None:
        del self._profiler
        self._profiler = None
