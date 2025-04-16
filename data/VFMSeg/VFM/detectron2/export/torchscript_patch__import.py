def _import(path):
    return _import_file('{}{}'.format(sys.modules[__name__].__name__,
        _counter), path, make_importable=True)
