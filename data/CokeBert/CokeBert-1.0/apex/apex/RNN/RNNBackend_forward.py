def forward(self, input):
    """
        forward()
        if not inited or bsz has changed this will create hidden states
        """
    self.init_hidden(input.size()[0])
    hidden_state = self.hidden[0] if self.n_hidden_states == 1 else self.hidden
    self.hidden = self.cell(input, hidden_state, self.w_ih, self.w_hh, b_ih
        =self.b_ih, b_hh=self.b_hh)
    if self.n_hidden_states > 1:
        self.hidden = list(self.hidden)
    else:
        self.hidden = [self.hidden]
    if self.output_size != self.hidden_size:
        self.hidden[0] = F.linear(self.hidden[0], self.w_ho)
    return tuple(self.hidden)
