def render(self, labels=True):
    self._run(render=True, labels=labels)
    return self.ims
