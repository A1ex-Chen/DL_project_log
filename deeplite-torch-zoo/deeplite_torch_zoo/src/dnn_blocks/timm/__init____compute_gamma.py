def _compute_gamma(self, x):
    return self.drop_prob / self.block_size ** 2
