def set_verbosity(verbosity: int) ->None:
    """
    Set the vebosity level for the 🤗 Transformers's root logger.

    Args:
        verbosity (:obj:`int`):
            Logging level, e.g., one of:

            - ``transformers.logging.CRITICAL`` or ``transformers.logging.FATAL``
            - ``transformers.logging.ERROR``
            - ``transformers.logging.WARNING`` or ``transformers.logging.WARN``
            - ``transformers.logging.INFO``
            - ``transformers.logging.DEBUG``
    """
    _configure_library_root_logger()
    _get_library_root_logger().setLevel(verbosity)
