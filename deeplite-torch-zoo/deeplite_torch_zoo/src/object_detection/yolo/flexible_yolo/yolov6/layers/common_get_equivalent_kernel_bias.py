def get_equivalent_kernel_bias(self):
    kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
    kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight)
    if self.rbr_avg is not None:
        kernelavg = self._avg_to_3x3_tensor(self.rbr_avg)
        kernel = kernel + kernelavg.to(self.rbr_1x1.weight.device)
    bias = bias3x3
    if self.rbr_identity is not None:
        input_dim = self.in_channels // self.groups
        kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=
            np.float32)
        for i in range(self.in_channels):
            kernel_value[i, i % input_dim, 1, 1] = 1
        id_tensor = torch.from_numpy(kernel_value).to(self.rbr_1x1.weight.
            device)
        kernel = kernel + id_tensor
    return kernel, bias
