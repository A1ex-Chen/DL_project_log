def is_pt_tf_cross_test(test_case):
    """
    Decorator marking a test as a test that control interactions between PyTorch and TensorFlow.

    PT+TF tests are skipped by default and we can run only them by setting RUN_PT_TF_CROSS_TESTS environment variable
    to a truthy value and selecting the is_pt_tf_cross_test pytest mark.

    """
    if not _run_pt_tf_cross_tests or not _torch_available or not _tf_available:
        return unittest.skip('test is PT+TF test')(test_case)
    else:
        try:
            import pytest
        except ImportError:
            return test_case
        else:
            return pytest.mark.is_pt_tf_cross_test()(test_case)
