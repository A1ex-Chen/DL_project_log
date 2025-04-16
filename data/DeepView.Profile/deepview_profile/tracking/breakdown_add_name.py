def add_name(self, name):
    if name in self._names:
        return
    self._names.append(name)
