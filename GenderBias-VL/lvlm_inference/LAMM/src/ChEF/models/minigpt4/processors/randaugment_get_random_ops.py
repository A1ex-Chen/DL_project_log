def get_random_ops(self):
    sampled_ops = np.random.choice(self.augs, self.N, replace=False)
    return [(op, self.M) for op in sampled_ops]
