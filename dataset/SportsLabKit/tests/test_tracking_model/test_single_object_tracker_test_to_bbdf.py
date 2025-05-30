def test_to_bbdf(self):
    sot = SingleObjectTracker()
    sot.update(detection=det0)
    sot.update(detection=det1)
    sot.update(detection=None)
    df = sot.to_bbdf()
    self.assertEqual(df.shape, (3, 5))
    self.assertEqual(df.iloc[0].tolist(), [10.0, 10.0, 5.0, 5.0, 0.9])
    self.assertEqual(df.iloc[1].tolist(), [20.0, 20.0, 3.0, 3.0, 0.75])
    self.assertEqual(df.iloc[2].tolist(), [20.0, 20.0, 3.0, 3.0, 0.75])
