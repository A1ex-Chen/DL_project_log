def fuse_bn_to_linear(bn_layer, linear_layer):
    LOGGER.info('Fusing BatchNorm layers of the teacher model')
    bn_st_dict = bn_layer.state_dict()
    conv_st_dict = linear_layer.state_dict()
    eps = bn_layer.eps
    mu = bn_st_dict['running_mean']
    var = bn_st_dict['running_var']
    gamma = bn_st_dict['weight']
    if 'bias' in bn_st_dict:
        beta = bn_st_dict['bias']
    else:
        beta = torch.zeros(gamma.size(0)).float().to(gamma.device)
    W = conv_st_dict['weight']
    if 'bias' in conv_st_dict:
        bias = conv_st_dict['bias']
    else:
        bias = torch.zeros(W.size(0)).float().to(gamma.device)
    denom = torch.sqrt(var + eps)
    b = beta - gamma.mul(mu).div(denom)
    A = gamma.div(denom)
    bias *= A
    A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)
    W.mul_(A)
    bias.add_(b)
    linear_layer.weight.data.copy_(W)
    if linear_layer.bias is None:
        linear_layer.bias = torch.nn.Parameter(bias)
    else:
        linear_layer.bias.data.copy_(bias)
