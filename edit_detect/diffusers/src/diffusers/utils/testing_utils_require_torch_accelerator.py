def require_torch_accelerator(test_case):
    """Decorator marking a test that requires an accelerator backend and PyTorch."""
    return unittest.skipUnless(is_torch_available() and torch_device !=
        'cpu', 'test requires accelerator+PyTorch')(test_case)
