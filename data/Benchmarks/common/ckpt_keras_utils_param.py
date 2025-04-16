def param(gParameters, key, dflt, type_=ParamType.STRING, allowed=None):
    """Pull key from parameters with type checks and conversions"""
    if key in gParameters:
        result = gParameters[key]
    else:
        if isinstance(dflt, ParamRequired):
            raise Exception("param key must be provided: '%s'" % key)
        result = dflt
    result = param_type_check(key, result, type_)
    param_allowed(key, result, allowed)
    return result
