def __init__(self, nasnet_a_large: nn.Module):
    super().__init__()
    self.reduction_cell_1 = nasnet_a_large.reduction_cell_1
    self.cell_12 = nasnet_a_large.cell_12
    self.cell_13 = nasnet_a_large.cell_13
    self.cell_14 = nasnet_a_large.cell_14
    self.cell_15 = nasnet_a_large.cell_15
    self.cell_16 = nasnet_a_large.cell_16
    self.cell_17 = nasnet_a_large.cell_17
