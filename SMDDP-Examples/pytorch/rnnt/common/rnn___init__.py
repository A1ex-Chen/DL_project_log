def __init__(self, input_size, hidden_size, num_layers, dropout,
    hidden_hidden_bias_scale, weights_init_scale, forget_gate_bias,
    multilayer_cudnn=True, **kwargs):
    super().__init__(num_layers)
    for i in range(num_layers):
        self.rnn.append(torch.nn.LSTM(input_size=input_size if i == 0 else
            hidden_size, hidden_size=hidden_size))
        self.rnn.append(torch.nn.Dropout(dropout))
    self.set_forgate_gate_bias(forget_gate_bias, hidden_size,
        hidden_hidden_bias_scale)
    for name, v in self.named_parameters():
        if 'weight' in name or 'bias' in name:
            v.data *= float(weights_init_scale)
    tensor_name = kwargs['tensor_name']
    logging.log_event(logging.constants.WEIGHTS_INITIALIZATION, metadata=
        dict(tensor=tensor_name))
