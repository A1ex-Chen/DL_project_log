@staticmethod
def _get_init_keys(cls):
    return set(dict(inspect.signature(cls.__init__).parameters).keys())
