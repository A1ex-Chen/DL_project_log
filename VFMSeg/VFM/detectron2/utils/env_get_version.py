def get_version(module, digit=2):
    return tuple(map(int, module.__version__.split('.')[:digit]))
