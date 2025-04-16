def __init__(self, keys, path, append=False):
    super(CSVLogger, self).__init__()
    self._keys = keys
    self._path = path
    if append is False or not os.path.exists(self._path):
        with open(self._path, 'w') as f:
            w = csv.DictWriter(f, self._keys)
            w.writeheader()
