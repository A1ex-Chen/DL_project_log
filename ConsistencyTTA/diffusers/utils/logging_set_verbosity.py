def set_verbosity(verbosity: int) ->None:
    """
    Set the verbosity level for the 🤗 Diffusers' root logger.

    Args:
        verbosity (`int`):
            Logging level, e.g., one of:

            - `diffusers.logging.CRITICAL` or `diffusers.logging.FATAL`
            - `diffusers.logging.ERROR`
            - `diffusers.logging.WARNING` or `diffusers.logging.WARN`
            - `diffusers.logging.INFO`
            - `diffusers.logging.DEBUG`
    """
    _configure_library_root_logger()
    _get_library_root_logger().setLevel(verbosity)
