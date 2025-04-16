def initialize(weights, initializer, kerasDefaults, seed=None, constant=0.0):
    if initializer == 'constant':
        return torch.nn.init.constant_(weights, val=constant)
    elif initializer == 'uniform':
        return torch.nn.init.uniform(weights, a=kerasDefaults[
            'minval_uniform'], b=kerasDefaults['maxval_uniform'])
    elif initializer == 'normal':
        return torch.nn.init.normal(weights, mean=kerasDefaults[
            'mean_normal'], std=kerasDefaults['stddev_normal'])
    elif initializer == 'glorot_normal':
        return torch.nn.init.xavier_normal(weights)
    elif initializer == 'glorot_uniform':
        return torch.nn.init.xavier_uniform_(weights)
    elif initializer == 'he_normal':
        return torch.nn.init.kaiming_uniform(weights)
