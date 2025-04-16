def load_from_fileobj(self, file, **kwargs):
    kwargs.setdefault('Loader', Loader)
    return yaml.load(file, **kwargs)
