def _find_caller():
    """
    Returns:
        str: module name of the caller
        tuple: a hashable key to be used to identify different callers
    """
    frame = sys._getframe(2)
    while frame:
        code = frame.f_code
        if os.path.join('utils', 'logger.') not in code.co_filename:
            mod_name = frame.f_globals['__name__']
            if mod_name == '__main__':
                mod_name = 'detectron2'
            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
        frame = frame.f_back
