def require_torchsde(test_case):
    """
    Decorator marking a test that requires torchsde. These tests are skipped when torchsde isn't installed.
    """
    return unittest.skipUnless(is_torchsde_available(),
        'test requires torchsde')(test_case)
