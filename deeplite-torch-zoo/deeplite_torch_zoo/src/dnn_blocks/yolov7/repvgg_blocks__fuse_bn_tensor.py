def _fuse_bn_tensor(self, branch):
    if branch is None:
        return 0, 0
    if isinstance(branch, nn.Sequential):
        kernel = branch[0].weight
        running_mean = branch[1].running_mean
        running_var = branch[1].running_var
        gamma = branch[1].weight
        beta = branch[1].bias
        eps = branch[1].eps
    else:
        assert isinstance(branch, nn.BatchNorm2d)
        if not hasattr(self, 'id_tensor'):
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            self.id_tensor = torch.from_numpy(kernel_value).to(branch.
                weight.device)
        kernel = self.id_tensor
        running_mean = branch.running_mean
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std
