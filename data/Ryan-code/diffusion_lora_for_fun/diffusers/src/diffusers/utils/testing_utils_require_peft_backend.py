def require_peft_backend(test_case):
    """
    Decorator marking a test that requires PEFT backend, this would require some specific versions of PEFT and
    transformers.
    """
    return unittest.skipUnless(USE_PEFT_BACKEND, 'test requires PEFT backend')(
        test_case)
