def _commit_removals(self):
    pop = self._pending_removals.pop
    d = self.data
    while True:
        try:
            key = pop()
        except IndexError:
            return
        try:
            del d[key]
        except KeyError:
            pass
