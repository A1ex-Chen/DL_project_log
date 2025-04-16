def load_from_fileobj(self, file, **kwargs):
    return pickle.load(file, **kwargs)
