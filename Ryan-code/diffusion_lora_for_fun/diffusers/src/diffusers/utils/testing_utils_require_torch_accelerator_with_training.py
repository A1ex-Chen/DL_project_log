def require_torch_accelerator_with_training(test_case):
    """Decorator marking a test that requires an accelerator with support for training."""
    return unittest.skipUnless(is_torch_available() and
        backend_supports_training(torch_device),
        'test requires accelerator with training support')(test_case)
