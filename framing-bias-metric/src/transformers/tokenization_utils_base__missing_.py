@classmethod
def _missing_(cls, value):
    raise ValueError('%r is not a valid %s, please select one of %s' % (
        value, cls.__name__, str(list(cls._value2member_map_.keys()))))
