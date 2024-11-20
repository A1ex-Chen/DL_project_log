def __init__(self, project_root):
    super().__init__()
    self._hook_manager = HookManager()
    self._module_parameters = WeakTensorKeyDictionary()
    self._project_root = project_root
