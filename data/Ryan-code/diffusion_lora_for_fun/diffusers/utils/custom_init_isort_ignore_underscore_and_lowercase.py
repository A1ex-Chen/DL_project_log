def ignore_underscore_and_lowercase(key: Callable[[Any], str]) ->Callable[[
    Any], str]:
    """
    Wraps a key function (as used in a sort) to lowercase and ignore underscores.
    """

    def _inner(x):
        return key(x).lower().replace('_', '')
    return _inner
