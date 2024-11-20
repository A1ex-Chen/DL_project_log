def setUp(self, max_abs_diff=0.001, max_rel_diff=1, iters=7):
    self.max_abs_diff = max_abs_diff
    self.max_rel_diff = max_rel_diff
    self.iters = iters
    torch.cuda.manual_seed(9876)
