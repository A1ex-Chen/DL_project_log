def is_pytest_running():
    """
    Determines whether pytest is currently running or not.

    Returns:
        (bool): True if pytest is running, False otherwise.
    """
    return ('PYTEST_CURRENT_TEST' in os.environ or 'pytest' in sys.modules or
        'pytest' in Path(ARGV[0]).stem)
