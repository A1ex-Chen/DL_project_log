def get_verbosity() ->int:
    """
    Return the current level for the 🤗 Diffusers' root logger as an int.

    Returns:
        `int`: The logging level.

    <Tip>

    🤗 Diffusers has following logging levels:

    - 50: `diffusers.logging.CRITICAL` or `diffusers.logging.FATAL`
    - 40: `diffusers.logging.ERROR`
    - 30: `diffusers.logging.WARNING` or `diffusers.logging.WARN`
    - 20: `diffusers.logging.INFO`
    - 10: `diffusers.logging.DEBUG`

    </Tip>"""
    _configure_library_root_logger()
    return _get_library_root_logger().getEffectiveLevel()
