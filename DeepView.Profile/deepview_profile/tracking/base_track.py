@contextlib.contextmanager
def track(self):
    self.start_tracking()
    try:
        yield self
    finally:
        self.stop_tracking()
