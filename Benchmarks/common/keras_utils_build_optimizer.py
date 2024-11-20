def build_optimizer(optimizer, lr, kerasDefaults):
    """Set the optimizer to the appropriate Keras optimizer function
    based on the input string and learning rate. Other required values
    are set to the Keras default values

    Parameters
    ----------
    optimizer : string
        String to choose the optimizer

        Options recognized: 'sgd', 'rmsprop', 'adagrad', adadelta', 'adam'
        See the Keras documentation for a full description of the options

    lr : float
        Learning rate

    kerasDefaults : list
        List of default parameter values to ensure consistency between frameworks

    Returns
    ----------
    The appropriate Keras optimizer function
    """
    if optimizer == 'sgd':
        return optimizers.SGD(lr=lr, decay=kerasDefaults['decay_lr'],
            momentum=kerasDefaults['momentum_sgd'], nesterov=kerasDefaults[
            'nesterov_sgd'])
    elif optimizer == 'rmsprop':
        return optimizers.RMSprop(lr=lr, rho=kerasDefaults['rho'], epsilon=
            kerasDefaults['epsilon'], decay=kerasDefaults['decay_lr'])
    elif optimizer == 'adagrad':
        return optimizers.Adagrad(lr=lr, epsilon=kerasDefaults['epsilon'],
            decay=kerasDefaults['decay_lr'])
    elif optimizer == 'adadelta':
        return optimizers.Adadelta(lr=lr, rho=kerasDefaults['rho'], epsilon
            =kerasDefaults['epsilon'], decay=kerasDefaults['decay_lr'])
    elif optimizer == 'adam':
        return optimizers.Adam(lr=lr, beta_1=kerasDefaults['beta_1'],
            beta_2=kerasDefaults['beta_2'], epsilon=kerasDefaults['epsilon'
            ], decay=kerasDefaults['decay_lr'])
