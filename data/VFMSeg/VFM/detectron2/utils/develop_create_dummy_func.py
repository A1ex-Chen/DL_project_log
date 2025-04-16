def create_dummy_func(func, dependency, message=''):
    """
    When a dependency of a function is not available, create a dummy function which throws
    ImportError when used.

    Args:
        func (str): name of the function.
        dependency (str or list[str]): name(s) of the dependency.
        message: extra message to print
    Returns:
        function: a function object
    """
    err = "Cannot import '{}', therefore '{}' is not available.".format(
        dependency, func)
    if message:
        err = err + ' ' + message
    if isinstance(dependency, (list, tuple)):
        dependency = ','.join(dependency)

    def _dummy(*args, **kwargs):
        raise ImportError(err)
    return _dummy
