def __init__(self, t=0.0):
    self.t = t
    self.cuda = torch.cuda.is_available()
