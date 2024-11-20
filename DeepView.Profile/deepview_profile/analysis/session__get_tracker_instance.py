def _get_tracker_instance(self):
    return Tracker(model_provider=self._model_provider, iteration_provider=
        self._iteration_provider, input_provider=self._input_provider,
        project_root=self._project_root, user_code_path=self.
        _path_to_entry_point_dir)
