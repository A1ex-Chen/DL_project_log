def before_train(self):
    for h in self._hooks:
        h.before_train()
