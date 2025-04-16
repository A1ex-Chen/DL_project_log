def LSTM(input_size, hidden_size, num_layers, bias=True, batch_first=False,
    dropout=0, bidirectional=False, output_size=None):
    """
    :class:`LSTM`
    """
    inputRNN = RNNCell(4, input_size, hidden_size, LSTMCell, 2, bias,
        output_size)
    return toRNNBackend(inputRNN, num_layers, bidirectional, dropout=dropout)
