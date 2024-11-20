def require_torch_gpu(test_case):
    """Decorator marking a test that requires CUDA and PyTorch. """
    if torch_device != 'cuda':
        return unittest.skip('test requires CUDA')(test_case)
    else:
        return test_case
