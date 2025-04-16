@contextlib.contextmanager
def ctx(self):
    yield self._ctx()
