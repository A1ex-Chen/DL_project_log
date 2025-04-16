@classmethod
def build_naming_info(cls):
    if cls.NAMING_INFO is not None:
        return
    info = dict(short_word={}, reverse_short_word={}, short_param={},
        reverse_short_param={})
    field_keys = list(cls.DEFAULTS.keys())
    for k in field_keys:
        cls.add_new_param_name(info, k)
    cls.NAMING_INFO = info
