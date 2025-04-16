def is_ubuntu() ->bool:
    """
    Check if the OS is Ubuntu.

    Returns:
        (bool): True if OS is Ubuntu, False otherwise.
    """
    with contextlib.suppress(FileNotFoundError):
        with open('/etc/os-release') as f:
            return 'ID=ubuntu' in f.read()
    return False
