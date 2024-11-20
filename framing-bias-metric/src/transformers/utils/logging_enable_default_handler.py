def enable_default_handler() ->None:
    """Enable the default handler of the HuggingFace Transformers's root logger."""
    _configure_library_root_logger()
    assert _default_handler is not None
    _get_library_root_logger().addHandler(_default_handler)
