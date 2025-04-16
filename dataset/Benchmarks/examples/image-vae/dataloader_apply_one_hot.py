def apply_one_hot(self, ch):
    return np.array(map(self.apply_t, ch))
