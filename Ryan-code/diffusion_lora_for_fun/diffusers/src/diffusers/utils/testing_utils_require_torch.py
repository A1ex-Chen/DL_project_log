def require_torch(test_case):
    """
    Decorator marking a test that requires PyTorch. These tests are skipped when PyTorch isn't installed.
    """
    return unittest.skipUnless(is_torch_available(), 'test requires PyTorch')(
        test_case)
