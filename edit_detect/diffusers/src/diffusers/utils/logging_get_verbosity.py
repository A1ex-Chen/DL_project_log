def get_verbosity() ->int:
    """
    Return the current level for the 🤗 Diffusers' root logger as an `int`.

    Returns:
        `int`:
            Logging level integers which can be one of:

            - `50`: `diffusers.logging.CRITICAL` or `diffusers.logging.FATAL`
            - `40`: `diffusers.logging.ERROR`
            - `30`: `diffusers.logging.WARNING` or `diffusers.logging.WARN`
            - `20`: `diffusers.logging.INFO`
            - `10`: `diffusers.logging.DEBUG`

    """
    _configure_library_root_logger()
    return _get_library_root_logger().getEffectiveLevel()
