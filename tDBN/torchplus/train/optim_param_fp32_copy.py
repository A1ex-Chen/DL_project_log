def param_fp32_copy(params):
    param_copy = [param.clone().type(torch.cuda.FloatTensor).detach() for
        param in params]
    for param in param_copy:
        param.requires_grad = True
    return param_copy
