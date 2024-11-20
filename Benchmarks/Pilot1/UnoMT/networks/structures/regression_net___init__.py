def __init__(self, encoder: nn.Module, input_dim: int, condition_dim: int,
    layer_dim: int, num_layers: int, activation: str):
    super(RgsNet, self).__init__()
    self.__encoder = encoder
    self.__rgs_net = nn.Sequential()
    prev_dim = input_dim + condition_dim
    for i in range(num_layers):
        self.__rgs_net.add_module('dense_%d' % i, nn.Linear(prev_dim,
            layer_dim))
        prev_dim = layer_dim
        self.__rgs_net.add_module('relu_%d' % i, nn.ReLU())
    self.__rgs_net.add_module('dense_%d' % num_layers, nn.Linear(prev_dim, 1))
    if activation.lower() == 'sigmoid':
        self.__rgs_net.add_module('activation', nn.Sigmoid())
    elif activation.lower() == 'tanh':
        self.__rgs_net.add_module('activation', nn.Tanh())
    else:
        pass
    self.__rgs_net.apply(basic_weight_init)
