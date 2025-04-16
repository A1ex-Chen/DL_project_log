@wraps(func)
def wrapper(self, *args, **kwargs):
    event_name = func.__name__
    self._invoke_callbacks(f'on_{event_name}_start')
    result = func(self, *args, **kwargs)
    self._invoke_callbacks(f'on_{event_name}_end')
    return result
