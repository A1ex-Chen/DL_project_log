def set_forgate_gate_bias(self, val, hidden_size, hidden_hidden_bias_scale):
    if val is not None:
        for name, v in self.rnn.named_parameters():
            if 'bias_ih' in name:
                idx, name = name.split('.', 1)
                bias = getattr(self.rnn[int(idx)], name)
                bias.data[hidden_size:2 * hidden_size].fill_(val)
            if 'bias_hh' in name:
                idx, name = name.split('.', 1)
                bias = getattr(self.rnn[int(idx)], name)
                bias.data[hidden_size:2 * hidden_size] *= float(
                    hidden_hidden_bias_scale)
