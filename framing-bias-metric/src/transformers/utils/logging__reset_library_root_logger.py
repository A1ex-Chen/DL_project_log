def _reset_library_root_logger() ->None:
    global _default_handler
    with _lock:
        if not _default_handler:
            return
        library_root_logger = _get_library_root_logger()
        library_root_logger.removeHandler(_default_handler)
        library_root_logger.setLevel(logging.NOTSET)
        _default_handler = None
