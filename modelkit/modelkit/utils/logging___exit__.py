def __exit__(self, exc_type, exc_value, exc_traceback):
    [contextvars.unbind_contextvars(key) for key in self._context]
    contextvars.bind_contextvars(**self._existing_vars)
