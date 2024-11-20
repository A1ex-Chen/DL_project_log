def test_load(self):
    codf = load_codf(csv_path)
    self.assertIsInstance(codf, CoordinatesDataFrame)
