def get_ubuntu_version():
    """
    Retrieve the Ubuntu version if the OS is Ubuntu.

    Returns:
        (str): Ubuntu version or None if not an Ubuntu OS.
    """
    if is_ubuntu():
        with contextlib.suppress(FileNotFoundError, AttributeError):
            with open('/etc/os-release') as f:
                return re.search('VERSION_ID="(\\d+\\.\\d+)"', f.read())[1]
