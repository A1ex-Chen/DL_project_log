def require_flax(test_case):
    """
    Decorator marking a test that requires JAX & Flax. These tests are skipped when one / both are not installed
    """
    return unittest.skipUnless(is_flax_available(), 'test requires JAX & Flax'
        )(test_case)
