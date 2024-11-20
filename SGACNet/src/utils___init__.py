def __init__(self, device):
    super(CrossEntropyLoss2dForValidDataUnweighted, self).__init__()
    self.ce_loss = nn.CrossEntropyLoss(weight=None, reduction='sum',
        ignore_index=-1)
    self.ce_loss.to(device)
    self.nr_pixels = 0
    self.total_loss = 0
