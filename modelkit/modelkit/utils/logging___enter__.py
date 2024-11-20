def __enter__(self):
    self._existing_vars = contextvars.merge_contextvars(logger=None,
        method_name=None, event_dict={})
    contextvars.bind_contextvars(**self._context)
    return self
