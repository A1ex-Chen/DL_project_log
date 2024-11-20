def test_cat_matches_widest(self):
    shape = self.b
    ys = [torch.randn(shape, dtype=torch.half) for _ in range(5)]
    x_float = torch.randn(shape)
    out = torch.cat(ys + [x_float])
    self.assertEqual(out.type(), FLOAT)
    x_half = torch.randn(shape, dtype=torch.half)
    out = torch.cat(ys + [x_half])
    self.assertEqual(out.type(), HALF)
