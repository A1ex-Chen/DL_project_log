def load(self, name):
    path = os.path.join(self.path, name)
    try:
        with open(path, encoding='utf-8') as fp:
            return self._load(fp)
    except FileNotFoundError:
        return self.DEFAULT_VALUE
