def __init__(self, N=2, M=10, isPIL=False, augs=[]):
    self.N = N
    self.M = M
    self.isPIL = isPIL
    if augs:
        self.augs = augs
    else:
        self.augs = list(arg_dict.keys())
