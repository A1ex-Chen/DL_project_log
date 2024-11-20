def __enter__(self):
    self.logger.addHandler(self.sh)
    return self
