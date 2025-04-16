def get_equivalent_kernel_bias(self):
    """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
    kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
    kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
    kernelid, biasid = self._fuse_bn_tensor(self.bn)
    return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1
        ) + kernelid, bias3x3 + bias1x1 + biasid
