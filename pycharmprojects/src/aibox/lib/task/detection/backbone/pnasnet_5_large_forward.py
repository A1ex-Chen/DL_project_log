def forward(self, x_cell_7: Tensor) ->Tensor:
    x_cell_8 = self.cell_8(x_cell_7, x_cell_7)
    x_cell_9 = self.cell_9(x_cell_7, x_cell_8)
    x_cell_10 = self.cell_10(x_cell_8, x_cell_9)
    x_cell_11 = self.cell_11(x_cell_9, x_cell_10)
    return x_cell_11
