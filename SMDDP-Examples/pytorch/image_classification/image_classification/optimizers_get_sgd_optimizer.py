def get_sgd_optimizer(parameters, lr, momentum, weight_decay, nesterov=
    False, bn_weight_decay=False):
    if bn_weight_decay:
        print(' ! Weight decay applied to BN parameters ')
        params = [v for n, v in parameters]
    else:
        print(' ! Weight decay NOT applied to BN parameters ')
        bn_params = [v for n, v in parameters if 'bn' in n]
        rest_params = [v for n, v in parameters if not 'bn' in n]
        print(len(bn_params))
        print(len(rest_params))
        params = [{'params': bn_params, 'weight_decay': 0}, {'params':
            rest_params, 'weight_decay': weight_decay}]
    optimizer = torch.optim.SGD(params, lr, momentum=momentum, weight_decay
        =weight_decay, nesterov=nesterov)
    return optimizer
