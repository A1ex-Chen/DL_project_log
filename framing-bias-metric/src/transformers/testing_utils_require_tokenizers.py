def require_tokenizers(test_case):
    """
    Decorator marking a test that requires 🤗 Tokenizers.

    These tests are skipped when 🤗 Tokenizers isn't installed.

    """
    if not _tokenizers_available:
        return unittest.skip('test requires tokenizers')(test_case)
    else:
        return test_case
