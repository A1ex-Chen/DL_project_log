def wrapper(self, *args, **kwargs):
    if hasattr(self, '_hf_hook') and hasattr(self._hf_hook, 'pre_forward'):
        self._hf_hook.pre_forward(self)
    return method(self, *args, **kwargs)
