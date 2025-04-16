def get_event_storage():
    """
    Returns:
        The :class:`EventStorage` object that's currently being used.
        Throws an error if no :class:`EventStorage` is currently enabled.
    """
    assert len(_CURRENT_STORAGE_STACK
        ), "get_event_storage() has to be called inside a 'with EventStorage(...)' context!"
    return _CURRENT_STORAGE_STACK[-1]
