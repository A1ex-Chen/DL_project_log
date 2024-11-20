def require_torch_multi_gpu(test_case):
    """
    Decorator marking a test that requires a multi-GPU setup (in PyTorch). These tests are skipped on a machine without
    multiple GPUs. To run *only* the multi_gpu tests, assuming all test names contain multi_gpu: $ pytest -sv ./tests
    -k "multi_gpu"
    """
    if not is_torch_available():
        return unittest.skip('test requires PyTorch')(test_case)
    import torch
    return unittest.skipUnless(torch.cuda.device_count() > 1,
        'test requires multiple GPUs')(test_case)
