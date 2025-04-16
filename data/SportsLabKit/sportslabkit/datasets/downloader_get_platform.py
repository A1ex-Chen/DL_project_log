def get_platform() ->str:
    """Get the platform of the current operating system.

    Returns:
        str: The platform of the current operating system, one of "linux", "mac", "windows", "other".
    """
    platforms = {'linux': 'linux', 'linux1': 'linux', 'linux2': 'linux',
        'darwin': 'mac', 'win32': 'windows'}
    if platform.system().lower() not in platforms:
        return 'other'
    return platforms[platform.system().lower()]
