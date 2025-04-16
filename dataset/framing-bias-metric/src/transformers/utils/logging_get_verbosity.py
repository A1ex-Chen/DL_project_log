def get_verbosity() ->int:
    """
    Return the current level for the 🤗 Transformers's root logger as an int.

    Returns:
        :obj:`int`: The logging level.

    .. note::

        🤗 Transformers has following logging levels:

        - 50: ``transformers.logging.CRITICAL`` or ``transformers.logging.FATAL``
        - 40: ``transformers.logging.ERROR``
        - 30: ``transformers.logging.WARNING`` or ``transformers.logging.WARN``
        - 20: ``transformers.logging.INFO``
        - 10: ``transformers.logging.DEBUG``
    """
    _configure_library_root_logger()
    return _get_library_root_logger().getEffectiveLevel()
