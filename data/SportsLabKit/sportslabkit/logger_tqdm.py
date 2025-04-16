def tqdm(*args, level: str='INFO', **kwargs) ->Iterable:
    """Wrapper for tqdm.tqdm that uses the logger's level.

    Args:
        *args: Arguments to pass to tqdm.tqdm
        **kwargs: Keyword arguments to pass to tqdm.tqdm
        level (str, optional): Logging level to set. Defaults to "INFO".

    Returns:
        Iterable: Iterable from tqdm progress bar
    """
    from tqdm import tqdm
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    enable = logger.level(LOG_LEVEL).no <= logger.level(level.upper()).no
    kwargs.update({'disable': not enable})
    return tqdm(*args, **kwargs)
