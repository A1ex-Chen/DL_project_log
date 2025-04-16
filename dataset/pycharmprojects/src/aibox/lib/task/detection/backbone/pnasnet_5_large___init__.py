def __init__(self, pnasnet_5_large: nn.Module):
    super().__init__()
    self.cell_8 = pnasnet_5_large.cell_8
    self.cell_9 = pnasnet_5_large.cell_9
    self.cell_10 = pnasnet_5_large.cell_10
    self.cell_11 = pnasnet_5_large.cell_11
