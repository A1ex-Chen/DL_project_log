def param_type_check_bool(key, value):
    if isinstance(value, bool):
        return value
    try:
        v = str2bool(value)
    except TypeError:
        raise TypeError("parameter: '%s' is '%s' but must be a %s" % key,
            str(value), str(ParamType.BOOLEAN))
    return v
