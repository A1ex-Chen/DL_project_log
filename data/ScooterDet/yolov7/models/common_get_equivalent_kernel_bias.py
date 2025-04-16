def get_equivalent_kernel_bias(self):
    kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
    kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
    kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
    return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1
        ) + kernelid, bias3x3 + bias1x1 + biasid
