def build_optimizer(model, optimizer, lr, kerasDefaults, trainable_only=True):
    if trainable_only:
        params = filter(lambda p: p.requires_grad, model.parameters())
    else:
        params = model.parameters()
    if optimizer == 'sgd':
        return torch.optim.GradientDescentMomentum(params, lr=lr,
            momentum_coef=kerasDefaults['momentum_sgd'], nesterov=
            kerasDefaults['nesterov_sgd'])
    elif optimizer == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr, alpha=
            kerasDefaults['rho'], eps=kerasDefaults['epsilon'])
    elif optimizer == 'adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=lr, eps=
            kerasDefaults['epsilon'])
    elif optimizer == 'adadelta':
        return torch.optim.Adadelta(params, eps=kerasDefaults['epsilon'],
            rho=kerasDefaults['rho'])
    elif optimizer == 'adam':
        return torch.optim.Adam(params, lr=lr, betas=[kerasDefaults[
            'beta_1'], kerasDefaults['beta_2']], eps=kerasDefaults['epsilon'])
