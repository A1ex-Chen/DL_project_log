def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith('linux'):
        return 'linux_x86_64'
    elif sys.platform == 'darwin':
        mac_version = '.'.join(platform.mac_ver()[0].split('.')[:2])
        return f'macosx_{mac_version}_x86_64'
    elif sys.platform == 'win32':
        return 'win_amd64'
    else:
        raise ValueError('Unsupported platform: {}'.format(sys.platform))
