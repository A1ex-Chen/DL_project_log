def before_step(self):
    self.storage.iter = self.iter
    for h in self._hooks:
        h.before_step()
