def __init__(self, linear_dim=False, vec_encode=False, z_fixed=-1.0,
    h_fixed=2.0):
    super().__init__()
    self.linear_dim = linear_dim
    self.z_fixed = z_fixed
    self.h_fixed = h_fixed
    self.vec_encode = vec_encode
