def check_version(current: str='0.0.0', minimum: str='0.0.0', name: str=
    'version ', pinned: bool=False, hard: bool=False, verbose: bool=False
    ) ->bool:
    """
    Check current version against the required minimum version.

    Args:
        current (str): Current version.
        minimum (str): Required minimum version.
        name (str): Name to be used in warning message.
        pinned (bool): If True, versions must match exactly. If False, minimum version must be satisfied.
        hard (bool): If True, raise an AssertionError if the minimum version is not met.
        verbose (bool): If True, print warning message if minimum version is not met.

    Returns:
        (bool): True if minimum version is met, False otherwise.
    """
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = current == minimum if pinned else current >= minimum
    warning_message = (
        f'WARNING ⚠️ {name}{minimum} is required by YOLOv8, but {name}{current} is currently installed'
        )
    if hard:
        assert result, warning_message
    if verbose and not result:
        LOGGER.warning(warning_message)
    return result
