def require_torch_tpu(test_case):
    """
    Decorator marking a test that requires a TPU (in PyTorch).
    """
    if not _torch_tpu_available:
        return unittest.skip('test requires PyTorch TPU')
    else:
        return test_case
