def param_type_check(key, value, type_):
    """
    Check that value is convertable to given type:
          if not, raise TypeError
    Return the value as converted to given type
    """
    if value is None:
        return value
    if type_ is ParamType.STRING:
        return str(value)
    if type_ is ParamType.BOOLEAN:
        return param_type_check_bool(key, value)
    if (type_ is ParamType.INTEGER or type_ is ParamType.INTEGER_NN or 
        type_ is ParamType.INTEGER_GZ):
        return param_type_check_int(key, value, type_)
    if type_ is ParamType.FLOAT or type_ is ParamType.FLOAT_NN:
        return param_type_check_float(key, value, type_)
    raise TypeError("param_type_check(): unknown type: '%s'" % str(type_))
