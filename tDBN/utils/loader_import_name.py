def import_name(name, package=None):
    module = importlib.import_module(name, package)
    return module
