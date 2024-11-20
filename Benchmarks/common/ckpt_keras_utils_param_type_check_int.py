def param_type_check_int(key, value, type_):
    if isinstance(value, int):
        result = value
    else:
        try:
            result = int(value)
        except TypeError:
            raise TypeError("parameter: '%s' is '%s' but must be a %s" % (
                key, str(value), str(type_)))
    if type_ == ParamType.INTEGER_NN:
        if result < 0:
            raise TypeError(("parameter: '%s' is '%s' " +
                'but must be non-negative') % (key, str(value)))
    if type_ == ParamType.INTEGER_GZ:
        if result <= 0:
            raise TypeError(("parameter: '%s' is '%s' " +
                'but must be greater-than-zero') % (key, str(value)))
    return result
