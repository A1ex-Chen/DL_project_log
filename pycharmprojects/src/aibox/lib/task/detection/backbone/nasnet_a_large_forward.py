def forward(self, x_cell_11: Tensor) ->Tensor:
    x_reduction_cell_1 = self.reduction_cell_1(x_cell_11, x_cell_11)
    x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_11)
    x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
    x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
    x_cell_15 = self.cell_15(x_cell_14, x_cell_13)
    x_cell_16 = self.cell_16(x_cell_15, x_cell_14)
    x_cell_17 = self.cell_17(x_cell_16, x_cell_15)
    return x_cell_17
