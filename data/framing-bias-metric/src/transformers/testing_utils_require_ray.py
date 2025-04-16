def require_ray(test_case):
    """
    Decorator marking a test that requires Ray/tune.

    These tests are skipped when Ray/tune isn't installed.

    """
    if not _has_ray:
        return unittest.skip('test requires Ray/tune')(test_case)
    else:
        return test_case
