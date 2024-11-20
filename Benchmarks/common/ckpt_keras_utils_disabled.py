def disabled(gParameters, key):
    """Is this parameter set to False?"""
    return key in gParameters and not gParameters[key]
