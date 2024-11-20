def inspect(*args, level: str='INFO', **kwargs) ->None:
    """Wrapper for rich.inspect that uses the logger's level.

    Args:
        *args: Arguments to pass to rich.inspect
        **kwargs: Keyword arguments to pass to rich.inspect
        level (str, optional): Logging level to set. Defaults to "INFO".
    """
    from rich import inspect
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    enable = logger.level(LOG_LEVEL).no <= logger.level(level.upper()).no
    if hasattr(args[0], __name__):
        if args[0].__name__ == 'inspect' and enable:
            inspect(inspect, *args[1:], **kwargs)
    elif enable:
        logger.log(level, f'Inspecting: {args}')
        inspect(*args, **kwargs)
