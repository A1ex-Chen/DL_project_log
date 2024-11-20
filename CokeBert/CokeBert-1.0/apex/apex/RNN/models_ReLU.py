def ReLU(input_size, hidden_size, num_layers, bias=True, batch_first=False,
    dropout=0, bidirectional=False, output_size=None):
    """
    :class:`ReLU`
    """
    inputRNN = RNNCell(1, input_size, hidden_size, RNNReLUCell, 1, bias,
        output_size)
    return toRNNBackend(inputRNN, num_layers, bidirectional, dropout=dropout)
