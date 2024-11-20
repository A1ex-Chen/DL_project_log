@contextlib.contextmanager
def disable_casts():
    _amp_state.handle._is_active = False
    yield
    _amp_state.handle._is_active = True
