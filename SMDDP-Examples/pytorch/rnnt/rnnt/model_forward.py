def forward(self, y):
    y = self.prediction['embed'](y)
    y = torch.nn.functional.pad(y, (0, 0, 1, 0))
    y = y.transpose(0, 1)
    bs = y.size(1)
    require_padding = bs < self.min_lstm_bs
    if require_padding:
        y = torch.nn.functional.pad(y, (0, 0, 0, self.min_lstm_bs - bs))
    g, hid = self.prediction['dec_rnn'](y, None)
    g = self.joint_pred(g.transpose(0, 1))
    if require_padding:
        g = g[:bs]
    return g
