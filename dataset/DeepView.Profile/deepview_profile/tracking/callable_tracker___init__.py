def __init__(self, hook_creator):
    super().__init__()
    self._hook_manager = HookManager()
    self._hook_creator = hook_creator
    self._torch_version = Version.parse_semantic_version(torch.__version__)
