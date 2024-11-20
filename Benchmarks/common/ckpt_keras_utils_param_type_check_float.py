def param_type_check_float(key, value, type_):
    if isinstance(value, float):
        result = value
    else:
        try:
            result = float(value)
        except TypeError:
            raise TypeError("parameter: '%s' is '%s' but must be a %s" % (
                key, str(value), str(type_)))
    if type_ == ParamType.FLOAT_NN:
        if result < 0:
            raise TypeError(("parameter: '%s' is '%s' " +
                'but must be non-negative') % (key, str(value)))
    return result
