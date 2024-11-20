def GRU(input_size, hidden_size, num_layers, bias=True, batch_first=False,
    dropout=0, bidirectional=False, output_size=None):
    """
    :class:`GRU`
    """
    inputRNN = RNNCell(3, input_size, hidden_size, GRUCell, 1, bias,
        output_size)
    return toRNNBackend(inputRNN, num_layers, bidirectional, dropout=dropout)
