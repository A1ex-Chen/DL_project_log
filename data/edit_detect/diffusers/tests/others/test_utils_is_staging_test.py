def is_staging_test(test_case):
    """
    Decorator marking a test as a staging test.

    Those tests will run using the staging environment of huggingface.co instead of the real model hub.
    """
    if not _run_staging:
        return unittest.skip('test is staging test')(test_case)
    else:
        return pytest.mark.is_staging_test()(test_case)
