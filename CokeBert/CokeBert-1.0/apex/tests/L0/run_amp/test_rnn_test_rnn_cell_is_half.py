def test_rnn_cell_is_half(self):
    cell = nn.RNNCell(self.h, self.h)
    self.run_cell_test(cell)
