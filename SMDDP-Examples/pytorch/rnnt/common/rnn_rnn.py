def rnn(input_size, hidden_size, num_layers, forget_gate_bias=1.0, dropout=
    0.0, decoupled=False, **kwargs):
    kwargs = dict(input_size=input_size, hidden_size=hidden_size,
        num_layers=num_layers, dropout=dropout, forget_gate_bias=
        forget_gate_bias, **kwargs)
    if decoupled:
        return DecoupledLSTM(**kwargs)
    else:
        return LSTM(**kwargs)
