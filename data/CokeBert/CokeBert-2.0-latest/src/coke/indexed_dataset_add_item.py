def add_item(self, tensor):
    bytes = self.out_file.write(np.array(tensor.numpy() + 1, dtype=self.dtype))
    self.data_offsets.append(self.data_offsets[-1] + bytes / self.element_size)
    for s in tensor.size():
        self.sizes.append(s)
    self.dim_offsets.append(self.dim_offsets[-1] + len(tensor.size()))
