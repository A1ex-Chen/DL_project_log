def init_lstm_(lstm, init_weight=0.1):
    """
    Initializes weights of LSTM layer.
    Weights and biases are initialized with uniform(-init_weight, init_weight)
    distribution.

    :param lstm: instance of torch.nn.LSTM
    :param init_weight: range for the uniform initializer
    """
    init.uniform_(lstm.weight_hh_l0.data, -init_weight, init_weight)
    init.uniform_(lstm.weight_ih_l0.data, -init_weight, init_weight)
    init.uniform_(lstm.bias_ih_l0.data, -init_weight, init_weight)
    init.zeros_(lstm.bias_hh_l0.data)
    if lstm.bidirectional:
        init.uniform_(lstm.weight_hh_l0_reverse.data, -init_weight, init_weight
            )
        init.uniform_(lstm.weight_ih_l0_reverse.data, -init_weight, init_weight
            )
        init.uniform_(lstm.bias_ih_l0_reverse.data, -init_weight, init_weight)
        init.zeros_(lstm.bias_hh_l0_reverse.data)
