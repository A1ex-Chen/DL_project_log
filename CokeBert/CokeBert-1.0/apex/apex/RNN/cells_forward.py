def forward(self, input):
    """
        mLSTMRNNCell.forward()
        """
    self.init_hidden(input.size()[0])
    hidden_state = self.hidden[0] if self.n_hidden_states == 1 else self.hidden
    self.hidden = list(self.cell(input, hidden_state, self.w_ih, self.w_hh,
        self.w_mih, self.w_mhh, b_ih=self.b_ih, b_hh=self.b_hh))
    if self.output_size != self.hidden_size:
        self.hidden[0] = F.linear(self.hidden[0], self.w_ho)
    return tuple(self.hidden)
