def parse_version(version='0.0.0') ->tuple:
    """
    Convert a version string to a tuple of integers, ignoring any extra non-numeric string attached to the version. This
    function replaces deprecated 'pkg_resources.parse_version(v)'.

    Args:
        version (str): Version string, i.e. '2.0.1+cpu'

    Returns:
        (tuple): Tuple of integers representing the numeric part of the version and the extra string, i.e. (2, 0, 1)
    """
    try:
        return tuple(map(int, re.findall('\\d+', version)[:3]))
    except Exception as e:
        LOGGER.warning(
            f'WARNING ⚠️ failure for parse_version({version}), returning (0, 0, 0): {e}'
            )
        return 0, 0, 0
