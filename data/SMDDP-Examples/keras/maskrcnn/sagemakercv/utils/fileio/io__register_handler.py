def _register_handler(handler, file_formats):
    """Register a handler for some file extensions.

    Args:
        handler (:obj:`BaseFileHandler`): Handler to be registered.
        file_formats (str or list[str]): File formats to be handled by this
            handler.
    """
    if not isinstance(handler, BaseFileHandler):
        raise TypeError('handler must be a child of BaseFileHandler, not {}'
            .format(type(handler)))
    if isinstance(file_formats, str):
        file_formats = [file_formats]
    if isinstance(file_formats, list):
        for f in file_formats:
            if not is_str(f):
                raise TypeError('file_formats must be a str or a list of str')
    for ext in file_formats:
        file_handlers[ext] = handler
