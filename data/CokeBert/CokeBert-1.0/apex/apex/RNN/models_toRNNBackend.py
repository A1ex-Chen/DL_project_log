def toRNNBackend(inputRNN, num_layers, bidirectional=False, dropout=0):
    """
    :class:`toRNNBackend`
    """
    if bidirectional:
        return bidirectionalRNN(inputRNN, num_layers, dropout=dropout)
    else:
        return stackedRNN(inputRNN, num_layers, dropout=dropout)
