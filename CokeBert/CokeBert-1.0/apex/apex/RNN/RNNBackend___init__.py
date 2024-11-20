def __init__(self, gate_multiplier, input_size, hidden_size, cell,
    n_hidden_states=2, bias=False, output_size=None):
    super(RNNCell, self).__init__()
    self.gate_multiplier = gate_multiplier
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.cell = cell
    self.bias = bias
    self.output_size = output_size
    if output_size is None:
        self.output_size = hidden_size
    self.gate_size = gate_multiplier * self.hidden_size
    self.n_hidden_states = n_hidden_states
    self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
    self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.output_size))
    if self.output_size != self.hidden_size:
        self.w_ho = nn.Parameter(torch.Tensor(self.output_size, self.
            hidden_size))
    self.b_ih = self.b_hh = None
    if self.bias:
        self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))
    self.hidden = [None for states in range(self.n_hidden_states)]
    self.reset_parameters()
