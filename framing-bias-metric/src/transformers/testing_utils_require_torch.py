def require_torch(test_case):
    """
    Decorator marking a test that requires PyTorch.

    These tests are skipped when PyTorch isn't installed.

    """
    if not _torch_available:
        return unittest.skip('test requires PyTorch')(test_case)
    else:
        return test_case
