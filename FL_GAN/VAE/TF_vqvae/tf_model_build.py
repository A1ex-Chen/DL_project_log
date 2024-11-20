def build(self, input_shape):
    self.conv.build(input_shape)
    kernel_shape = self.conv.kernel.get_shape()
    self.mask = np.zeros(shape=kernel_shape)
    self.mask[:kernel_shape[0] // 2, ...] = 1.0
    self.mask[kernel_shape[0] // 2, :kernel_shape[1] // 2, ...] = 1.0
    if self.mask_type == 'B':
        self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0
