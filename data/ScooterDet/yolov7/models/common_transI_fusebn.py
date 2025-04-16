def transI_fusebn(kernel, bn):
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    return kernel * (gamma / std).reshape(-1, 1, 1, 1
        ), bn.bias - bn.running_mean * gamma / std
