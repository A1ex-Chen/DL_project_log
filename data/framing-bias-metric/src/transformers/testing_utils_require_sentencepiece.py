def require_sentencepiece(test_case):
    """
    Decorator marking a test that requires SentencePiece.

    These tests are skipped when SentencePiece isn't installed.

    """
    if not _sentencepiece_available:
        return unittest.skip('test requires SentencePiece')(test_case)
    else:
        return test_case
