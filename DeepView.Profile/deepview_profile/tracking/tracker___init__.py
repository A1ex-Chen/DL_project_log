def __init__(self, model_provider, iteration_provider, input_provider,
    project_root, user_code_path):
    self._model_provider = model_provider
    self._iteration_provider = iteration_provider
    self._input_provider = input_provider
    self._project_root = project_root
    self._user_code_path = user_code_path
    self._tracker_state = _TrackerState.CREATED
    self._model = None
    self._weight_tracker = None
    self._activations_tracker = None
    self._peak_usage_bytes = None
    self._operation_tracker = None
