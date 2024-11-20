def test_tqdm(self):
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        with open(tmpdir / 'test.txt', 'w') as f:
            for i in tqdm(range(10), level='DEBUG'):
                f.write(str(i))
        self.assertTrue((tmpdir / 'test.txt').exists())
