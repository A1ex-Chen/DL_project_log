def after_step(self):
    for h in self._hooks:
        h.after_step()
