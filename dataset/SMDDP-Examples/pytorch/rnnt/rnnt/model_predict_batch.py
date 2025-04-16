def predict_batch(self, y, state=None, add_sos=True):
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
    y_embed = self.prediction['embed'](abs(y.unsqueeze(1)))
    mask = y == -1
    mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.pred_n_hid)
    y_embed_masked = y_embed.masked_fill_(mask, 0)
    y_embed_masked = y_embed_masked.transpose(0, 1)
    g, hid = self.prediction['dec_rnn'](y_embed_masked, state)
    g = self.joint_pred(g.transpose(0, 1))
    return g, hid
