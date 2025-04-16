@TryExcept('Showing images is not supported in this environment')
def show(self, labels=True):
    self._run(show=True, labels=labels)
