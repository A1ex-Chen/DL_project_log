def mockenv(**kwargs):
    """
    this is a convenience wrapper, that allows this:

    @mockenv(RUN_SLOW=True, USE_TF=False) def test_something(): run_slow = os.getenv("RUN_SLOW", False) use_tf =
    os.getenv("USE_TF", False)
    """
    return unittest.mock.patch.dict(os.environ, kwargs)
