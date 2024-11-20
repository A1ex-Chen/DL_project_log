@functools.wraps(func)
def wrapper(self, *args, **kwargs):
    with ExitStack() as stack:
        if hasattr(self, 'profiler'):
            stack.enter_context(self.profiler.profile(self.configuration_key))
        vals = func(self, *args, **kwargs)
        yield from vals
