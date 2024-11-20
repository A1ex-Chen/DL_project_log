def get_optimizer(params, lr, betas, eps, momentum, optimizer_name):
    if optimizer_name.lower() == 'adamw':
        optimizer = optim.AdamW(params, lr=lr, betas=betas, eps=eps)
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(params, lr=lr, momentum=momentum)
    elif optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(params, lr=lr, betas=betas, eps=eps)
    else:
        raise ValueError('optimizer name is not correct')
    return optimizer
