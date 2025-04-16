def python39_available():
    major, minor = get_python_version()
    return major == 3 and minor >= 9
