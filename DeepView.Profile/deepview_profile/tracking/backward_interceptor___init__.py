def __init__(self):
    self._backward_hooks = HookManager()
    self.backward_root = None
