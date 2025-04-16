def _configure_library_root_logger() ->None:
    global _default_handler
    with _lock:
        if _default_handler:
            return
        _default_handler = logging.StreamHandler()
        _default_handler.flush = sys.stderr.flush
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())
        library_root_logger.propagate = False
