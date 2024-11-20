def check_version(current: str='0.0.0', required: str='0.0.0', name: str=
    'version', hard: bool=False, verbose: bool=False, msg: str='') ->bool:
    """
    Check current version against the required version or range.

    Args:
        current (str): Current version or package name to get version from.
        required (str): Required version or range (in pip-style format).
        name (str, optional): Name to be used in warning message.
        hard (bool, optional): If True, raise an AssertionError if the requirement is not met.
        verbose (bool, optional): If True, print warning message if requirement is not met.
        msg (str, optional): Extra message to display if verbose.

    Returns:
        (bool): True if requirement is met, False otherwise.

    Example:
        ```python
        # Check if current version is exactly 22.04
        check_version(current='22.04', required='==22.04')

        # Check if current version is greater than or equal to 22.04
        check_version(current='22.10', required='22.04')  # assumes '>=' inequality if none passed

        # Check if current version is less than or equal to 22.04
        check_version(current='22.04', required='<=22.04')

        # Check if current version is between 20.04 (inclusive) and 22.04 (exclusive)
        check_version(current='21.10', required='>20.04,<22.04')
        ```
    """
    if not current:
        LOGGER.warning(
            f'WARNING ⚠️ invalid check_version({current}, {required}) requested, please check values.'
            )
        return True
    elif not current[0].isdigit():
        try:
            name = current
            current = metadata.version(current)
        except metadata.PackageNotFoundError as e:
            if hard:
                raise ModuleNotFoundError(emojis(
                    f'WARNING ⚠️ {current} package is required but not installed'
                    )) from e
            else:
                return False
    if not required:
        return True
    op = ''
    version = ''
    result = True
    c = parse_version(current)
    for r in required.strip(',').split(','):
        op, version = re.match('([^0-9]*)([\\d.]+)', r).groups()
        v = parse_version(version)
        if op == '==' and c != v:
            result = False
        elif op == '!=' and c == v:
            result = False
        elif op in {'>=', ''} and not c >= v:
            result = False
        elif op == '<=' and not c <= v:
            result = False
        elif op == '>' and not c > v:
            result = False
        elif op == '<' and not c < v:
            result = False
    if not result:
        warning = (
            f'WARNING ⚠️ {name}{op}{version} is required, but {name}=={current} is currently installed {msg}'
            )
        if hard:
            raise ModuleNotFoundError(emojis(warning))
        if verbose:
            LOGGER.warning(warning)
    return result
