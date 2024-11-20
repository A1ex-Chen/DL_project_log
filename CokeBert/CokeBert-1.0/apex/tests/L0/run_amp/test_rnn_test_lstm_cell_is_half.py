def test_lstm_cell_is_half(self):
    cell = nn.LSTMCell(self.h, self.h)
    self.run_cell_test(cell, state_tuple=True)
