def __init__(self, project_root):
    super().__init__()
    self._callable_tracker = CallableTracker(self._callable_hook_creator)
    self._project_root = project_root
    self.grad_function_contexts = {}
    self._processing_hook = False
