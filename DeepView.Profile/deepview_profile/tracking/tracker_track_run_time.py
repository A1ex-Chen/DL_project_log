def track_run_time(self):
    if self._tracker_state == _TrackerState.CREATED:
        with user_code_environment(self._user_code_path, self._project_root):
            self._model = self._model_provider()
    elif self._tracker_state != _TrackerState.MEMORY_TRACKED:
        raise RuntimeError('Run time tracking has already been performed.')
    self._tracker_state = _TrackerState.RUN_TIME_TRACKED
    with user_code_environment(self._user_code_path, self._project_root):
        inputs = self._input_provider()
        iteration = self._iteration_provider(self._model)
    print('Tracking both input and iteration')
    self._operation_tracker = OperationRunTimeTracker(self._project_root)
    backward_interceptor = BackwardInterceptor()
    with self._operation_tracker.track():
        with backward_interceptor.intercept():
            with user_code_environment(self._user_code_path, self._project_root
                ):
                inputs = self._input_provider()
                iteration(*inputs)
