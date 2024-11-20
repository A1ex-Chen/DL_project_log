def call_hook(self, fn_name):
    for hook in self._hooks:
        getattr(hook, fn_name)(self)
