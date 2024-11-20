def __init__(self, N=2, M=10, p=0.0, tensor_in_tensor_out=True, augs=[]):
    self.N = N
    self.M = M
    self.p = p
    self.tensor_in_tensor_out = tensor_in_tensor_out
    if augs:
        self.augs = augs
    else:
        self.augs = list(arg_dict.keys())
