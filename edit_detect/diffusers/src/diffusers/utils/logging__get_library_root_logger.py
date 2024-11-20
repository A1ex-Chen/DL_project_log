def _get_library_root_logger() ->logging.Logger:
    return logging.getLogger(_get_library_name())
