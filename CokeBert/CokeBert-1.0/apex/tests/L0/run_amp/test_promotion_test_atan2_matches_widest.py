def test_atan2_matches_widest(self):
    fns = [lambda x, y: torch.atan2(x, y), lambda x, y: x.atan2(y)]
    self.run_binary_promote_test(fns, (self.b,))
