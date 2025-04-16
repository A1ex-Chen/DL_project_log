def check_pip_update_available():
    """
    Checks if a new version of the ultralytics package is available on PyPI.

    Returns:
        (bool): True if an update is available, False otherwise.
    """
    if ONLINE and IS_PIP_PACKAGE:
        with contextlib.suppress(Exception):
            from ultralytics import __version__
            latest = check_latest_pypi_version()
            if check_version(__version__, f'<{latest}'):
                LOGGER.info(
                    f"New https://pypi.org/project/ultralytics/{latest} available 😃 Update with 'pip install -U ultralytics'"
                    )
                return True
    return False
