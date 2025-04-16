def test_inspect(self):
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        with open(tmpdir / 'test.txt', 'w') as f:
            inspect(f, level='DEBUG')
        self.assertTrue((tmpdir / 'test.txt').exists())
