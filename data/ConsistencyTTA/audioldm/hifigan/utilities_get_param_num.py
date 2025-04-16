def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param
