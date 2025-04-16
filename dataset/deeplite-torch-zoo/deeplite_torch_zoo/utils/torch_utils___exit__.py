def __exit__(self, type, value, traceback):
    """
        Stop timing.
        """
    self.dt = self.time() - self.start
    self.t += self.dt
