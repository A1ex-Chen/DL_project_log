def require_faiss(test_case):
    """Decorator marking a test that requires faiss."""
    if not _faiss_available:
        return unittest.skip('test requires `faiss`')(test_case)
    else:
        return test_case
