def nightly(test_case):
    """
    Decorator marking a test that runs nightly in the diffusers CI.

    Slow tests are skipped by default. Set the RUN_NIGHTLY environment variable to a truthy value to run them.

    """
    return unittest.skipUnless(_run_nightly_tests, 'test is nightly')(test_case
        )
