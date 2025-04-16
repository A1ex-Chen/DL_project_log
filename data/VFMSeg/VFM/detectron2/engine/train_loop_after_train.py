def after_train(self):
    self.storage.iter = self.iter
    for h in self._hooks:
        h.after_train()
