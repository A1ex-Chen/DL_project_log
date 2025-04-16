def _flatten_parameters(self):
    for layer in self.rnn:
        if isinstance(layer, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)):
            layer._flatten_parameters()
