def test_bce_raises_by_default(self):
    assertion = lambda fn, x: self.assertRaises(NotImplementedError, fn, x)
    self.bce_common(assertion)
