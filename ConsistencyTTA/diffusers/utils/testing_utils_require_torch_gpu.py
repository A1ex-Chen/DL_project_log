def require_torch_gpu(test_case):
    """Decorator marking a test that requires CUDA and PyTorch."""
    return unittest.skipUnless(is_torch_available() and torch_device ==
        'cuda', 'test requires PyTorch+CUDA')(test_case)
