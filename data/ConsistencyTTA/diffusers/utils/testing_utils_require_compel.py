def require_compel(test_case):
    """
    Decorator marking a test that requires compel: https://github.com/damian0815/compel. These tests are skipped when
    the library is not installed.
    """
    return unittest.skipUnless(is_compel_available(), 'test requires compel')(
        test_case)
