def __init__(self, project_root, entry_point, path_to_entry_point_dir,
    model_provider, input_provider, iteration_provider, batch_size,
    entry_point_static_analyzer):
    self._project_root = project_root
    self._entry_point = entry_point
    self._path_to_entry_point_dir = path_to_entry_point_dir
    self._model_provider = model_provider
    self._input_provider = input_provider
    self._iteration_provider = iteration_provider
    self._batch_size = batch_size
    self._entry_point_static_analyzer = entry_point_static_analyzer
    self._profiler = None
    self._memory_usage_percentage = None
    self._batch_size_iteration_run_time_ms = None
    self._batch_size_peak_usage_bytes = None
    self._utilization = None
    self._energy_table_interface = EnergyTableInterface(DatabaseInterface()
        .connection)
