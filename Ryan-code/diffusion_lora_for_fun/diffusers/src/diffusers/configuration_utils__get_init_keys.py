@staticmethod
def _get_init_keys(input_class):
    return set(dict(inspect.signature(input_class.__init__).parameters).keys())
