def init_matrix(self, state_size):
    self.matrix = np.zeros((1, state_size, state_size, self.vocab_size),
        dtype=np.uint8)
