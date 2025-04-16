def perhaps_convert_float(param, total):
    if isinstance(param, float):
        param = int(param * total)
    return param
