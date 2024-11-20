def track_memory(self):
    if self._tracker_state != _TrackerState.CREATED:
        raise RuntimeError(
            'Memory tracking must be the first operation performed on a new Tracker.'
            )
    self._tracker_state = _TrackerState.MEMORY_TRACKED
    initial_memory_bytes = torch.cuda.memory_allocated()
    if initial_memory_bytes != 0:
        logger.debug(
            'Non-zero initial memory usage during memory tracking: %d bytes',
            initial_memory_bytes)
    self._weight_tracker = WeightsTracker(self._project_root)
    with user_code_environment(self._user_code_path, self._project_root):
        with self._weight_tracker.track():
            self._model = self._model_provider()
        iteration = self._iteration_provider(self._model)
        iteration(*self._input_provider())
    self._activations_tracker = ActivationsTracker(self._project_root)
    self._activations_tracker.track_memory_usage(iteration, self.
        _input_provider, self._user_code_path)
    torch.cuda.reset_max_memory_allocated()
    with user_code_environment(self._user_code_path, self._project_root):
        iteration(*self._input_provider())
    self._peak_usage_bytes = torch.cuda.max_memory_allocated()
