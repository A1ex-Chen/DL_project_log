def disable_propagation() ->None:
    """
    Disable propagation of the library log outputs. Note that log propagation is disabled by default.
    """
    _configure_library_root_logger()
    _get_library_root_logger().propagate = False
