def __init__(self, gnmt, loss_fn):
    super().__init__()
    self.gnmt = gnmt
    self.loss_fn = loss_fn
