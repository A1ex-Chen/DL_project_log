def __exit__(self, exc_type, exc_value, traceback):
    shutil.rmtree(self.name)
