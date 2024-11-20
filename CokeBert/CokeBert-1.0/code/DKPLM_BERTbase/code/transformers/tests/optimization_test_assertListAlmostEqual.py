def assertListAlmostEqual(self, list1, list2, tol):
    self.assertEqual(len(list1), len(list2))
    for a, b in zip(list1, list2):
        self.assertAlmostEqual(a, b, delta=tol)
