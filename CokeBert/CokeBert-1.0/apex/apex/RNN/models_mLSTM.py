def mLSTM(input_size, hidden_size, num_layers, bias=True, batch_first=False,
    dropout=0, bidirectional=False, output_size=None):
    """
    :class:`mLSTM`
    """
    inputRNN = mLSTMRNNCell(input_size, hidden_size, bias=bias, output_size
        =output_size)
    return toRNNBackend(inputRNN, num_layers, bidirectional, dropout=dropout)
