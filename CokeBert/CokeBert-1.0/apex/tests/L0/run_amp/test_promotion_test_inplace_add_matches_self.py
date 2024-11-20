def test_inplace_add_matches_self(self):
    fn = lambda x, y: x.add_(y)
    self.run_binary_promote_test([fn], (self.b,), x_inplace=True)
