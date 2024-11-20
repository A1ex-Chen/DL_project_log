def build_activation(activation):
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'sigmoid':
        return torch.nn.Sigmoid()
    elif activation == 'tanh':
        return torch.nn.Tanh()
