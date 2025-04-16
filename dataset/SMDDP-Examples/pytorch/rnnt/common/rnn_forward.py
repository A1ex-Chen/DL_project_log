def forward(self, x, hx=None):
    assert len(self.rnn) > 0, 'RNN not initialized'
    hx = self._parse_hidden_state(hx)
    hs = []
    cs = []
    rnn_idx = 0
    for layer in self.rnn:
        if isinstance(layer, torch.nn.Dropout):
            x = layer(x)
        else:
            x, h_out = layer(x, hx[rnn_idx])
            hs.append(h_out[0])
            cs.append(h_out[1])
            rnn_idx += 1
            del h_out
    h_0 = torch.cat(hs, dim=0)
    c_0 = torch.cat(cs, dim=0)
    return x, (h_0, c_0)
