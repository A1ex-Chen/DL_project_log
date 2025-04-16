def get_tests_dir(append_path=None):
    """
    Args:
        append_path: optional path to append to the tests dir path
    Return:
        The full path to the `tests` dir, so that the tests can be invoked from anywhere. Optionally `append_path` is
        joined after the `tests` dir the former is provided.
    """
    caller__file__ = inspect.stack()[1][1]
    tests_dir = os.path.abspath(os.path.dirname(caller__file__))
    while not tests_dir.endswith('tests'):
        tests_dir = os.path.dirname(tests_dir)
    if append_path:
        return os.path.join(tests_dir, append_path)
    else:
        return tests_dir
