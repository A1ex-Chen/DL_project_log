def measure(iterations):
    with user_code_environment(self._path_to_entry_point_dir, self.
        _project_root):
        self._start_event.record()
        for _ in range(iterations):
            self._iteration(*inputs)
        self._end_event.record()
    torch.cuda.synchronize()
    return self._start_event.elapsed_time(self._end_event)
