def require_flax(test_case):
    """
    Decorator marking a test that requires JAX & Flax

    These tests are skipped when one / both are not installed

    """
    if not _flax_available:
        test_case = unittest.skip('test requires JAX & Flax')(test_case)
    return test_case
