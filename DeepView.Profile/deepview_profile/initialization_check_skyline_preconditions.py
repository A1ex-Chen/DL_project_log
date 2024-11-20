def check_skyline_preconditions(args):
    """
    This is the first function that should run before importing any other
    DeepView code.
    """
    _configure_logging(args)
    if not _validate_dependencies():
        sys.exit(1)
    if not _validate_gpu():
        sys.exit(1)
