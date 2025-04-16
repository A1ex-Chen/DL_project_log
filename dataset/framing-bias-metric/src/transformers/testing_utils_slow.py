def slow(test_case):
    """
    Decorator marking a test as slow.

    Slow tests are skipped by default. Set the RUN_SLOW environment variable to a truthy value to run them.

    """
    if not _run_slow_tests:
        return unittest.skip('test is slow')(test_case)
    else:
        return test_case
