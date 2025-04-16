def _get_survival_prob(self, block_id):
    drop_rate = 1.0 - self.survival_prob
    sp = 1.0 - drop_rate * float(block_id) / self.num_blocks
    return sp
