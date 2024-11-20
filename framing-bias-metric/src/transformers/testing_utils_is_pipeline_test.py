def is_pipeline_test(test_case):
    """
    Decorator marking a test as a pipeline test.

    Pipeline tests are skipped by default and we can run only them by setting RUN_PIPELINE_TESTS environment variable
    to a truthy value and selecting the is_pipeline_test pytest mark.

    """
    if not _run_pipeline_tests:
        return unittest.skip('test is pipeline test')(test_case)
    else:
        try:
            import pytest
        except ImportError:
            return test_case
        else:
            return pytest.mark.is_pipeline_test()(test_case)
