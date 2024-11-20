def calc_abn_activation(ABN_layer):
    activation = nn.Identity()
    if isinstance(ABN_layer, inplace_abn.ABN):
        if ABN_layer.activation == 'relu':
            activation = nn.ReLU(inplace=True)
        elif ABN_layer.activation == 'leaky_relu':
            activation = nn.LeakyReLU(negative_slope=ABN_layer.
                activation_param, inplace=True)
        elif ABN_layer.activation == 'elu':
            activation = nn.ELU(alpha=ABN_layer.activation_param, inplace=True)
    return activation
