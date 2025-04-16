def get(self, extension):
    if extension not in self._registry:
        raise RuntimeError(f'Missing extension {self._name}/{extension}')
    return self._registry[extension]
