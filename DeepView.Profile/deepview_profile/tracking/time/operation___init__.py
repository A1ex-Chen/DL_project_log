def __init__(self, project_root):
    super().__init__()
    self._callable_tracker = CallableTracker(self._hook_creator)
    self._profiler = OperationProfiler()
    self._project_root = project_root
    self._processing_hook = False
    self.operations = []
