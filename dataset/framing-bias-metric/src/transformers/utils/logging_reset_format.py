def reset_format() ->None:
    """
    Resets the formatting for HuggingFace Transformers's loggers.

    All handlers currently bound to the root logger are affected by this method.
    """
    handlers = _get_library_root_logger().handlers
    for handler in handlers:
        handler.setFormatter(None)
