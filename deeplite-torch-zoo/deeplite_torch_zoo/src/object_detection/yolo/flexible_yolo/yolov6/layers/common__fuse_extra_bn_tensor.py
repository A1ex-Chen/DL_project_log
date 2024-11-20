def _fuse_extra_bn_tensor(self, kernel, bias, branch):
    assert isinstance(branch, nn.BatchNorm2d)
    running_mean = branch.running_mean - bias
    running_var = branch.running_var
    gamma = branch.weight
    beta = branch.bias
    eps = branch.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std
