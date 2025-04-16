def get_logger(name: Optional[str]=None) ->logging.Logger:
    """
    Return a logger with the specified name.

    This function is not supposed to be directly accessed unless you are writing a custom diffusers module.
    """
    if name is None:
        name = _get_library_name()
    _configure_library_root_logger()
    return logging.getLogger(name)
