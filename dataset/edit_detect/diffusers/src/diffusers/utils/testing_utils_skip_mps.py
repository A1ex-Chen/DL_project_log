def skip_mps(test_case):
    """Decorator marking a test to skip if torch_device is 'mps'"""
    return unittest.skipUnless(torch_device != 'mps',
        "test requires non 'mps' device")(test_case)
