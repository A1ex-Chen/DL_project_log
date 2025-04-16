def add_handler(handler: logging.Handler) ->None:
    """adds a handler to the HuggingFace Diffusers' root logger."""
    _configure_library_root_logger()
    assert handler is not None
    _get_library_root_logger().addHandler(handler)
