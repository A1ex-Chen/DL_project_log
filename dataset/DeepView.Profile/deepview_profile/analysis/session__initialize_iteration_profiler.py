def _initialize_iteration_profiler(self):
    self._profiler = IterationProfiler.new_from(self._model_provider, self.
        _input_provider, self._iteration_provider, self.
        _path_to_entry_point_dir, self._project_root)
