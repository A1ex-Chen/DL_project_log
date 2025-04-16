def __enter__(self):
    self.name = tempfile.mkdtemp()
    return self.name
