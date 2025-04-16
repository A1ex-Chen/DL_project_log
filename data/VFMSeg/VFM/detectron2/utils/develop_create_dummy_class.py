def create_dummy_class(klass, dependency, message=''):
    """
    When a dependency of a class is not available, create a dummy class which throws ImportError
    when used.

    Args:
        klass (str): name of the class.
        dependency (str): name of the dependency.
        message: extra message to print
    Returns:
        class: a class object
    """
    err = "Cannot import '{}', therefore '{}' is not available.".format(
        dependency, klass)
    if message:
        err = err + ' ' + message


    class _DummyMetaClass(type):

        def __getattr__(_, __):
            raise ImportError(err)


    class _Dummy(object, metaclass=_DummyMetaClass):

        def __init__(self, *args, **kwargs):
            raise ImportError(err)
    return _Dummy
