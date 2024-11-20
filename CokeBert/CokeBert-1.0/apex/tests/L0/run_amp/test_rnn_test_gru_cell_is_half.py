def test_gru_cell_is_half(self):
    cell = nn.GRUCell(self.h, self.h)
    self.run_cell_test(cell)
