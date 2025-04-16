def require_torch_accelerator_with_fp16(test_case):
    """Decorator marking a test that requires an accelerator with support for the FP16 data type."""
    return unittest.skipUnless(_is_torch_fp16_available(torch_device),
        'test requires accelerator with fp16 support')(test_case)
