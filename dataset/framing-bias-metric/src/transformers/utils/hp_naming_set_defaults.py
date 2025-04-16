@classmethod
def set_defaults(cls, prefix, defaults):
    cls.PREFIX = prefix
    cls.DEFAULTS = defaults
    cls.build_naming_info()
