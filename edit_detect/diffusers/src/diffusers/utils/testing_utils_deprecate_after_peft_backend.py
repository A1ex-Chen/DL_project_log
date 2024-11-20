def deprecate_after_peft_backend(test_case):
    """
    Decorator marking a test that will be skipped after PEFT backend
    """
    return unittest.skipUnless(not USE_PEFT_BACKEND,
        'test skipped in favor of PEFT backend')(test_case)
