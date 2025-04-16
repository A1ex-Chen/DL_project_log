@classmethod
def shortname(cls, params):
    cls.build_naming_info()
    assert cls.PREFIX is not None
    name = [copy.copy(cls.PREFIX)]
    for k, v in params.items():
        if k not in cls.DEFAULTS:
            raise Exception(
                f'You should provide a default value for the param name {k} with value {v}'
                )
        if v == cls.DEFAULTS[k]:
            continue
        key = cls.NAMING_INFO['short_param'][k]
        if isinstance(v, bool):
            v = 1 if v else 0
        sep = '' if isinstance(v, (int, float)) else '-'
        e = f'{key}{sep}{v}'
        name.append(e)
    return '_'.join(name)
