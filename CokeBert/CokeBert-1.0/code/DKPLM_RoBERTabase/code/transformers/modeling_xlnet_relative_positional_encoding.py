def relative_positional_encoding(self, qlen, klen, bsz=None):
    """create relative positional encoding."""
    freq_seq = torch.arange(0, self.d_model, 2.0, dtype=torch.float)
    inv_freq = 1 / torch.pow(10000, freq_seq / self.d_model)
    if self.attn_type == 'bi':
        beg, end = klen, -qlen
    elif self.attn_type == 'uni':
        beg, end = klen, -1
    else:
        raise ValueError('Unknown `attn_type` {}.'.format(self.attn_type))
    if self.bi_data:
        fwd_pos_seq = torch.arange(beg, end, -1.0, dtype=torch.float)
        bwd_pos_seq = torch.arange(-beg, -end, 1.0, dtype=torch.float)
        if self.clamp_len > 0:
            fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
            bwd_pos_seq = bwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
        if bsz is not None:
            fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, 
                bsz // 2)
            bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, 
                bsz // 2)
        else:
            fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
            bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)
        pos_emb = torch.cat([fwd_pos_emb, bwd_pos_emb], dim=1)
    else:
        fwd_pos_seq = torch.arange(beg, end, -1.0)
        if self.clamp_len > 0:
            fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
        pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)
    pos_emb = pos_emb.to(next(self.parameters()))
    return pos_emb
