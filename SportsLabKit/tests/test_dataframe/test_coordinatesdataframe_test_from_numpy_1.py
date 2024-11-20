def test_from_numpy_1(self):
    arr = np.random.rand(10, 22, 2)
    codf = CoordinatesDataFrame.from_numpy(arr)
    self.assertIsInstance(codf, CoordinatesDataFrame)
    self.assertEqual(codf.shape, (10, 44))
