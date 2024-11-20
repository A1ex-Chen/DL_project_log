def add_connect(self, from_state, to_state, w_group):
    assert self.matrix is not None
    for w_index in w_group:
        self.matrix[0, from_state, to_state, w_index] = 1
        self.matrix[0, from_state, from_state, w_index] = 0
