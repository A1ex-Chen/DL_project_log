def _fuse_bn_tensor(self, branch) ->Tuple[torch.Tensor, torch.Tensor]:
    """Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
    if isinstance(branch, nn.Sequential):
        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps
    else:
        assert isinstance(branch, nn.BatchNorm2d)
        if not hasattr(self, 'id_tensor'):
            input_dim = self.in_channels // self.groups
            kernel_value = torch.zeros((self.in_channels, input_dim, self.
                kernel_size, self.kernel_size), dtype=branch.weight.dtype,
                device=branch.weight.device)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, self.kernel_size // 2, self.
                    kernel_size // 2] = 1
            self.id_tensor = kernel_value
        kernel = self.id_tensor
        running_mean = branch.running_mean
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std
