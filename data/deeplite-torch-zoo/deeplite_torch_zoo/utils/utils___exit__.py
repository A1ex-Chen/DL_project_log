def __exit__(self, type, value, traceback):
    self.dt = self.time() - self.start
    self.t += self.dt
