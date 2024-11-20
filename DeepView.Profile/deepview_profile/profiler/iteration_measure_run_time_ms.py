def measure_run_time_ms(self, batch_size, initial_repetitions=None):
    """
        Measures the iteration run time in milliseconds.

        NOTE: This method will raise a RuntimeError if there is not enough GPU
              memory to run the iteration.
        """
    with user_code_environment(self._path_to_entry_point_dir, self.
        _project_root):
        inputs = self._input_provider(batch_size=batch_size)
        for _ in range(10):
            self._iteration(*inputs)
    torch.cuda.synchronize()

    def measure(iterations):
        with user_code_environment(self._path_to_entry_point_dir, self.
            _project_root):
            self._start_event.record()
            for _ in range(iterations):
                self._iteration(*inputs)
            self._end_event.record()
        torch.cuda.synchronize()
        return self._start_event.elapsed_time(self._end_event)
    repetitions = 3 if initial_repetitions is None else initial_repetitions
    max_repetitions = 50 if initial_repetitions is None else max(50,
        initial_repetitions)
    torch.cuda.reset_max_memory_allocated()
    lesser = measure(repetitions) / repetitions
    peak_usage_bytes = torch.cuda.max_memory_allocated()
    min_profile_time_ms = 2000
    max_repetitions = max(max_repetitions, min_profile_time_ms / lesser)
    logger.debug('Iters: %d, Measured: %f', repetitions, lesser)
    while repetitions <= max_repetitions:
        doubled = repetitions * 2
        greater = measure(doubled) / doubled
        logger.debug('Iters: %d, Measured: %f (range: %f)', doubled,
            greater, max(lesser, greater) / min(lesser, greater))
        if max(lesser, greater) / min(lesser, greater) < 1.05:
            break
        repetitions = doubled
        lesser = greater
    return min(lesser, greater), peak_usage_bytes, repetitions
