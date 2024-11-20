def require_note_seq(test_case):
    """
    Decorator marking a test that requires note_seq. These tests are skipped when note_seq isn't installed.
    """
    return unittest.skipUnless(is_note_seq_available(),
        'test requires note_seq')(test_case)
