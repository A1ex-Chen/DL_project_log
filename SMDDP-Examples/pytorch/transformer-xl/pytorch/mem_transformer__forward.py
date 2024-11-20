def _forward(self, dec_inp, mems=None):
    qlen, bsz = dec_inp.size()
    word_emb = self.word_emb(dec_inp)
    mlen = mems[0].size(0) if mems is not None else 0
    klen = mlen + qlen
    if self.same_length:
        all_ones = word_emb.new_ones(qlen, klen)
        mask_len = klen - self.mem_len - 1
        if mask_len > 0:
            mask_shift_len = qlen - mask_len
        else:
            mask_shift_len = qlen
        dec_attn_mask = (torch.triu(all_ones, 1 + mlen) + torch.tril(
            all_ones, -mask_shift_len)).bool()
    else:
        dec_attn_mask = torch.triu(word_emb.new_ones(qlen, klen), diagonal=
            1 + mlen).bool()
    hids = []
    if self.attn_type == 0:
        pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
            dtype=word_emb.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)
        core_out = self.drop(word_emb)
        pos_emb = self.drop(pos_emb)
        for i, layer in enumerate(self.layers):
            hids.append(core_out.detach())
            mems_i = None if mems is None else mems[i]
            core_out = layer(core_out, pos_emb, self.r_w_bias, self.
                r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
    elif self.attn_type == 1:
        core_out = self.drop(word_emb)
        for i, layer in enumerate(self.layers):
            hids.append(core_out.detach())
            if self.clamp_len > 0:
                r_emb = self.r_emb[i][-self.clamp_len:]
                r_bias = self.r_bias[i][-self.clamp_len:]
            else:
                r_emb, r_bias = self.r_emb[i], self.r_bias[i]
            mems_i = None if mems is None else mems[i]
            core_out = layer(core_out, r_emb, self.r_w_bias[i], r_bias,
                dec_attn_mask=dec_attn_mask, mems=mems_i)
    elif self.attn_type == 2:
        pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
            dtype=word_emb.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)
        core_out = self.drop(word_emb + pos_emb[-qlen:])
        for i, layer in enumerate(self.layers):
            hids.append(core_out.detach())
            mems_i = None if mems is None else mems[i]
            if mems_i is not None and len(mems_i) and i == 0:
                mems_i += pos_emb[:mlen]
            core_out = layer(core_out, dec_attn_mask=dec_attn_mask, mems=mems_i
                )
    elif self.attn_type == 3:
        core_out = self.drop(word_emb)
        for i, layer in enumerate(self.layers):
            hids.append(core_out.detach())
            mems_i = None if mems is None else mems[i]
            if mems_i is not None and len(mems_i) and mlen > 0:
                cur_emb = self.r_emb[i][:-qlen]
                cur_size = cur_emb.size(0)
                if cur_size < mlen:
                    cur_emb_pad = cur_emb[0:1].expand(mlen - cur_size, -1, -1)
                    cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                else:
                    cur_emb = cur_emb[-mlen:]
                mems_i += cur_emb.view(mlen, 1, -1)
            core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)
            core_out = layer(core_out, dec_attn_mask=dec_attn_mask, mems=mems_i
                )
    core_out = self.drop(core_out)
    new_mems = self._update_mems(hids, mems, qlen, mlen)
    return core_out, new_mems
