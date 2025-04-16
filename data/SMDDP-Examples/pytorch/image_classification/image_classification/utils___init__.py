def __init__(self, sig=signal.SIGTERM):
    self.sig = sig
    self.device = torch.device('cuda')
