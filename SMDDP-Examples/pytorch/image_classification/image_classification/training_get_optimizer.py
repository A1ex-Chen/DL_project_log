def get_optimizer(parameters, lr, args, state=None):
    if args.optimizer == 'sgd':
        optimizer = get_sgd_optimizer(parameters, lr, momentum=args.
            momentum, weight_decay=args.weight_decay, nesterov=args.
            nesterov, bn_weight_decay=args.bn_weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = get_rmsprop_optimizer(parameters, lr, alpha=args.
            rmsprop_alpha, momentum=args.momentum, weight_decay=args.
            weight_decay, eps=args.rmsprop_eps, bn_weight_decay=args.
            bn_weight_decay)
    if not state is None:
        optimizer.load_state_dict(state)
    return optimizer
