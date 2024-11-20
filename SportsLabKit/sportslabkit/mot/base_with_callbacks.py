def with_callbacks(func):
    """Decorator for wrapping methods that require callback invocations.

    Args:
        func (callable): The method to wrap.

    Returns:
        callable: The wrapped method.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        event_name = func.__name__
        self._invoke_callbacks(f'on_{event_name}_start')
        result = func(self, *args, **kwargs)
        self._invoke_callbacks(f'on_{event_name}_end')
        return result
    return wrapper
