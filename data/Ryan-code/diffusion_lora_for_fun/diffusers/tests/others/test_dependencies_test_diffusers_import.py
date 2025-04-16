def test_diffusers_import(self):
    try:
        import diffusers
    except ImportError:
        assert False
