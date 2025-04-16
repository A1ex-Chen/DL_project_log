def require_python39_or_higher(test_case):

    def python39_available():
        major, minor = get_python_version()
        return major == 3 and minor >= 9
    return unittest.skipUnless(python39_available(),
        'test requires Python 3.9 or higher')(test_case)
