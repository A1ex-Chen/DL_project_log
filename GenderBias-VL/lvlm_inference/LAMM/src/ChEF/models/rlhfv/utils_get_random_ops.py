def get_random_ops(self):
    sampled_ops = np.random.choice(self.augs, self.N)
    return [(op, 0.5, self.M) for op in sampled_ops]
