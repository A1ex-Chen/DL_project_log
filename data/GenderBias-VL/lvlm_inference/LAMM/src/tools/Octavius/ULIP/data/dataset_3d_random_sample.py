def random_sample(self, pc, num):
    np.random.shuffle(self.permutation)
    pc = pc[self.permutation[:num]]
    return pc
