def disable_default_handler() ->None:
    """Disable the default handler of the HuggingFace Transformers's root logger."""
    _configure_library_root_logger()
    assert _default_handler is not None
    _get_library_root_logger().removeHandler(_default_handler)
