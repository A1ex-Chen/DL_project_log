def require_torch_2(test_case):
    """
    Decorator marking a test that requires PyTorch 2. These tests are skipped when it isn't installed.
    """
    return unittest.skipUnless(is_torch_available() and is_torch_version(
        '>=', '2.0.0'), 'test requires PyTorch 2')(test_case)
