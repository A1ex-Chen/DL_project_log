def _forward(self, dec_inp, mems: torch.Tensor):
    qlen, bsz = dec_inp.size()
    word_emb = self.word_emb(dec_inp)
    mlen = mems[0].size(0) if mems is not None else 0
    klen = mlen + qlen
    all_ones = torch.ones((qlen, klen), device=torch.device('cuda'), dtype=
        self.dtype)
    if self.same_length:
        mask_len = klen - self.mem_len - 1
        if mask_len > 0:
            mask_shift_len = qlen - mask_len
        else:
            mask_shift_len = qlen
        dec_attn_mask = (torch.triu(all_ones, 1 + mlen) + torch.tril(
            all_ones, -mask_shift_len)).to(torch.bool)
    else:
        dec_attn_mask = torch.triu(all_ones, diagonal=1 + mlen).to(torch.bool)
    pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
        dtype=word_emb.dtype)
    if self.clamp_len > 0:
        pos_seq.clamp_(max=self.clamp_len)
    pos_emb = self.pos_emb(pos_seq)
    core_out = self.drop(word_emb)
    pos_emb = self.drop(pos_emb)
    hids = []
    for i, layer in enumerate(self.layers):
        hids.append(core_out)
        mems_i = None if mems is None else mems[i]
        core_out = layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias,
            dec_attn_mask=dec_attn_mask, mems=mems_i)
    core_out = self.drop(core_out)
    new_mems = self._update_mems(hids, mems, qlen, mlen)
    return core_out, new_mems
