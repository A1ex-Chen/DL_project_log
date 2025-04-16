def predict(self, y, state=None, add_sos=True):
    """
        B - batch size
        U - label length
        H - Hidden dimension size
        L - Number of decoder layers = 2

        Args:
            y: (B, U)

        Returns:
            Tuple (g, hid) where:
                g: (B, U + 1, H)
                hid: (h, c) where h is the final sequence hidden state and c is
                    the final cell state:
                        h (tensor), shape (L, B, H)
                        c (tensor), shape (L, B, H)
        """
    if y is not None:
        y = self.prediction['embed'](y)
    else:
        B = 1 if state is None else state[0].size(1)
        y = torch.zeros((B, 1, self.pred_n_hid)).to(device=self.joint_enc.
            weight.device, dtype=self.joint_enc.weight.dtype)
    if add_sos:
        y = torch.nn.functional.pad(y, (0, 0, 1, 0))
    y = y.transpose(0, 1)
    bs = y.size(1)
    require_padding = bs < self.min_lstm_bs
    if require_padding:
        y = torch.nn.functional.pad(y, (0, 0, 0, self.min_lstm_bs - bs))
    g, hid = self.prediction['dec_rnn'](y, state)
    g = g.transpose(0, 1)
    if require_padding:
        g = g[:bs]
    g = self.joint_pred(g)
    del y, state
    return g, hid
