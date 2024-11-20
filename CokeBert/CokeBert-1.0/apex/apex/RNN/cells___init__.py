def __init__(self, input_size, hidden_size, bias=False, output_size=None):
    gate_multiplier = 4
    super(mLSTMRNNCell, self).__init__(gate_multiplier, input_size,
        hidden_size, mLSTMCell, n_hidden_states=2, bias=bias, output_size=
        output_size)
    self.w_mih = nn.Parameter(torch.Tensor(self.output_size, self.input_size))
    self.w_mhh = nn.Parameter(torch.Tensor(self.output_size, self.output_size))
    self.reset_parameters()
