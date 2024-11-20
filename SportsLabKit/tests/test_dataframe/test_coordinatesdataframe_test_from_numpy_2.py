def test_from_numpy_2(self):
    arr = np.random.rand(10, 23, 2)
    codf = CoordinatesDataFrame.from_numpy(arr)
    self.assertIsInstance(codf, CoordinatesDataFrame)
    self.assertEqual(codf.shape, (10, 46))
