def enable_propagation() ->None:
    """
    Enable propagation of the library log outputs. Please disable the HuggingFace Transformers's default handler to
    prevent double logging if the root logger has been configured.
    """
    _configure_library_root_logger()
    _get_library_root_logger().propagate = True
