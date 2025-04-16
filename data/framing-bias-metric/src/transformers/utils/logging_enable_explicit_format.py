def enable_explicit_format() ->None:
    """
    Enable explicit formatting for every HuggingFace Transformers's logger. The explicit formatter is as follows:

    ::

        [LEVELNAME|FILENAME|LINE NUMBER] TIME >> MESSAGE

    All handlers currently bound to the root logger are affected by this method.
    """
    handlers = _get_library_root_logger().handlers
    for handler in handlers:
        formatter = logging.Formatter(
            '[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s'
            )
        handler.setFormatter(formatter)
