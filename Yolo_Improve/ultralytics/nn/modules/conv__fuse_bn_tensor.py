def _fuse_bn_tensor(self, branch):
    """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
    if branch is None:
        return 0, 0
    if isinstance(branch, Conv):
        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps
    elif isinstance(branch, nn.BatchNorm2d):
        if not hasattr(self, 'id_tensor'):
            input_dim = self.c1 // self.g
            kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.
                float32)
            for i in range(self.c1):
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
