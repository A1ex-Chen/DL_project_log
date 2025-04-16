def enabled(gParameters, key):
    """Is this parameter set to True?"""
    return key in gParameters and gParameters[key]
