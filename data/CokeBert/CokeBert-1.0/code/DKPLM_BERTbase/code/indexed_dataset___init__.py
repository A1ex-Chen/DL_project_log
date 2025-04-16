def __init__(self, out_file, dtype=np.int32):
    self.out_file = open(out_file, 'wb')
    self.dtype = dtype
    self.data_offsets = [0]
    self.dim_offsets = [0]
    self.sizes = []
    self.element_size = self.element_sizes[self.dtype]
