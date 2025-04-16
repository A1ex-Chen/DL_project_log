def remove_handler(handler: logging.Handler) ->None:
    """removes given handler from the HuggingFace Diffusers' root logger."""
    _configure_library_root_logger()
    assert handler is not None and handler in _get_library_root_logger(
        ).handlers
    _get_library_root_logger().removeHandler(handler)
