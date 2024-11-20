def param_allowed(key, value, allowed):
    """
    Check that the value is in the list of allowed values
    If allowed is None, there is no check, simply success
    """
    if allowed is None:
        return
    if value not in allowed:
        raise ValueError(("hyperparameter '%s'='%s' is not in the " +
            'list of allowed values: %s') % (key, value, str(allowed)))
