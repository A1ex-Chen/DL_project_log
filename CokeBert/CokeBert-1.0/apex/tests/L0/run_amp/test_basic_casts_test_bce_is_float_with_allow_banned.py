def test_bce_is_float_with_allow_banned(self):
    self.handle._deactivate()
    self.handle = amp.init(enabled=True, allow_banned=True)
    assertion = lambda fn, x: self.assertEqual(fn(x).type(), FLOAT)
    self.bce_common(assertion)
