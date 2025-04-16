def require_optuna(test_case):
    """
    Decorator marking a test that requires optuna.

    These tests are skipped when optuna isn't installed.

    """
    if not _has_optuna:
        return unittest.skip('test requires optuna')(test_case)
    else:
        return test_case
