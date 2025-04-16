def init_row(self, state_index):
    assert self.matrix is not None
    self.matrix[0, state_index, state_index, :] = 1
