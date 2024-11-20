def _get_kernel_bias(self) ->Tuple[torch.Tensor, torch.Tensor]:
    """Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
    kernel_scale = 0
    bias_scale = 0
    if self.rbr_scale is not None:
        kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
        pad = self.kernel_size // 2
        kernel_scale = torch.nn.functional.pad(kernel_scale, [pad, pad, pad,
            pad])
    kernel_identity = 0
    bias_identity = 0
    if self.rbr_skip is not None:
        kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)
    kernel_conv = 0
    bias_conv = 0
    for ix in range(self.num_conv_branches):
        _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
        kernel_conv += _kernel
        bias_conv += _bias
    kernel_final = kernel_conv + kernel_scale + kernel_identity
    bias_final = bias_conv + bias_scale + bias_identity
    return kernel_final, bias_final
