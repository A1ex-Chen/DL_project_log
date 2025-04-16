def require_torch_accelerator_with_fp64(test_case):
    """Decorator marking a test that requires an accelerator with support for the FP64 data type."""
    return unittest.skipUnless(_is_torch_fp64_available(torch_device),
        'test requires accelerator with fp64 support')(test_case)
