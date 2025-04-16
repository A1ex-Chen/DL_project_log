def _wrap_target(target):
    """Wrap the MNIST target in a dictionary

    The multitask classifier of DARTS expects a
    dictionary of target tasks. Here we simply wrap
    MNIST's target in a dictionary.
    """
    return {'digits': target}
