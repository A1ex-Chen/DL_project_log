def joint(self, f, g, apex_transducer_joint=None, f_len=None,
    dict_meta_data=None):
    """
        f should be shape (B, T, H)
        g should be shape (B, U + 1, H)

        returns:
            logits of shape (B, T, U, K + 1)
        """
    if apex_transducer_joint is None:
        f = f.unsqueeze(dim=2)
        g = g.unsqueeze(dim=1)
        h = f + g
        B, T, U, H = h.size()
        res = self.joint_net(h.view(-1, H))
        res = res.view(B, T, U, -1)
    else:
        h = self.my_transducer_joint(f, g, f_len, dict_meta_data['g_len'],
            dict_meta_data['batch_offset'], dict_meta_data['packed_batch'])
        res = self.joint_net(h)
    del f, g
    return res
