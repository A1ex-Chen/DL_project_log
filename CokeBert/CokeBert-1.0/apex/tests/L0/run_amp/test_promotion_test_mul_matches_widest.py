def test_mul_matches_widest(self):
    fns = [lambda x, y: torch.mul(x, y), lambda x, y: x.mul(y)]
    self.run_binary_promote_test(fns, (self.b,))
