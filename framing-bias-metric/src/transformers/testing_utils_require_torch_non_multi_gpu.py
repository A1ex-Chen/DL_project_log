def require_torch_non_multi_gpu(test_case):
    """
    Decorator marking a test that requires 0 or 1 GPU setup (in PyTorch).
    """
    if not _torch_available:
        return unittest.skip('test requires PyTorch')(test_case)
    import torch
    if torch.cuda.device_count() > 1:
        return unittest.skip('test requires 0 or 1 GPU')(test_case)
    else:
        return test_case
