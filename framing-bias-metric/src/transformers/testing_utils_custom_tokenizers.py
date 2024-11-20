def custom_tokenizers(test_case):
    """
    Decorator marking a test for a custom tokenizer.

    Custom tokenizers require additional dependencies, and are skipped by default. Set the RUN_CUSTOM_TOKENIZERS
    environment variable to a truthy value to run them.
    """
    if not _run_custom_tokenizers:
        return unittest.skip('test of custom tokenizers')(test_case)
    else:
        return test_case
