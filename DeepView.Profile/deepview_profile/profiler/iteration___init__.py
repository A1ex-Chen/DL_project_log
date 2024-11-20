def __init__(self, iteration, input_provider, path_to_entry_point_dir,
    project_root):
    self._iteration = iteration
    self._input_provider = input_provider
    self._path_to_entry_point_dir = path_to_entry_point_dir
    self._project_root = project_root
    self._start_event = torch.cuda.Event(enable_timing=True)
    self._end_event = torch.cuda.Event(enable_timing=True)
