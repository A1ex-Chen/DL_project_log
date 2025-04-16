def patcher(record: dict[str, str | dict[str, str]]) ->dict[str, str | dict
    [str, str]]:
    """Customize loguru's log format.

    See the Loguru docs for details on `record` here, https://loguru.readthedocs.io/en/stable/api/logger.html.

    Args:
        record (Dict): Loguru record

    Returns:
        Dict: Loguru record
    """
    if record.get('function') == '<module>':
        if is_interactive():
            record['function'] = 'IPython'
        else:
            record['function'] = 'Python'
    if record['extra'].get('classname'):
        record['extra']['classname'] += ':'
    return record
