def test_inplace_exp_is_error_for_half(self):
    xs = torch.randn(self.b)
    xs.exp_()
    self.assertEqual(xs.type(), FLOAT)
    xs = torch.randn(self.b, dtype=torch.half)
    with self.assertRaises(NotImplementedError):
        xs.exp_()
