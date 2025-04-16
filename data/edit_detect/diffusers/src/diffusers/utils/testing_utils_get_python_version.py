def get_python_version():
    sys_info = sys.version_info
    major, minor = sys_info.major, sys_info.minor
    return major, minor
