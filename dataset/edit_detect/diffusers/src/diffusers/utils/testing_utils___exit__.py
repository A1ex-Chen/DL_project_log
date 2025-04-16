def __exit__(self, *exc):
    self.logger.removeHandler(self.sh)
    self.out = self.io.getvalue()
