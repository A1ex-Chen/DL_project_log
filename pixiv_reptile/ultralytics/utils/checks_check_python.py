def check_python(minimum: str='3.8.0', hard: bool=True) ->bool:
    """
    Check current python version against the required minimum version.

    Args:
        minimum (str): Required minimum version of python.
        hard (bool, optional): If True, raise an AssertionError if the requirement is not met.

    Returns:
        (bool): Whether the installed Python version meets the minimum constraints.
    """
    return check_version(PYTHON_VERSION, minimum, name='Python', hard=hard)
