def require_tf(test_case):
    """
    Decorator marking a test that requires TensorFlow.

    These tests are skipped when TensorFlow isn't installed.

    """
    if not _tf_available:
        return unittest.skip('test requires TensorFlow')(test_case)
    else:
        return test_case
