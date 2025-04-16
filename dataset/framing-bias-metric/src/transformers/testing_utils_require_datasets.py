def require_datasets(test_case):
    """Decorator marking a test that requires datasets."""
    if not _datasets_available:
        return unittest.skip('test requires `datasets`')(test_case)
    else:
        return test_case
