def __init__(self, blank_idx, precision, packed_input):
    super().__init__()
    self.t_loss = TransducerLoss(packed_input=packed_input)
    self.blank_idx = blank_idx
    self.precision = precision
