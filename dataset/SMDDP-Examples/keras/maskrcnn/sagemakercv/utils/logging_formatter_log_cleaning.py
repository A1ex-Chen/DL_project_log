def log_cleaning(hide_deprecation_warnings=False):
    if hide_deprecation_warnings:
        warnings.simplefilter('ignore')
        from tensorflow.python.util import deprecation
        from tensorflow.python.util import deprecation_wrapper
        deprecation._PRINT_DEPRECATION_WARNINGS = False
        deprecation_wrapper._PER_MODULE_WARNING_LIMIT = 0
    formatter = _logging.Formatter('[%(levelname)s] %(message)s')
    from tensorflow.python.platform import tf_logging
    tf_logging.get_logger().propagate = False
    _logging.getLogger().propagate = False
    for handler in _logging.getLogger().handlers:
        handler.setFormatter(formatter)
