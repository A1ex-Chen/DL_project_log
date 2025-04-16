def create_mask(self, qlen, mlen):
    """
        Creates causal attention mask. Float mask where 1.0 indicates masked, 0.0 indicates not-masked.

        Args:
            qlen: TODO Lysandre didn't fill
            mlen: TODO Lysandre didn't fill

        ::

                  same_length=False:      same_length=True:
                  <mlen > <  qlen >       <mlen > <  qlen >
               ^ [0 0 0 0 0 1 1 1 1]     [0 0 0 0 0 1 1 1 1]
                 [0 0 0 0 0 0 1 1 1]     [1 0 0 0 0 0 1 1 1]
            qlen [0 0 0 0 0 0 0 1 1]     [1 1 0 0 0 0 0 1 1]
                 [0 0 0 0 0 0 0 0 1]     [1 1 1 0 0 0 0 0 1]
               v [0 0 0 0 0 0 0 0 0]     [1 1 1 1 0 0 0 0 0]

        """
    attn_mask = torch.ones([qlen, qlen])
    mask_up = torch.triu(attn_mask, diagonal=1)
    attn_mask_pad = torch.zeros([qlen, mlen])
    ret = torch.cat([attn_mask_pad, mask_up], dim=1)
    if self.same_length:
        mask_lo = torch.tril(attn_mask, diagonal=-1)
        ret = torch.cat([ret[:, :qlen] + mask_lo, ret[:, qlen:]], dim=1)
    ret = ret.to(next(self.parameters()))
    return ret
