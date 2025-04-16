def relative_positional_encoding(self, qlen, klen, bsz=None, dtype=None):
    """create relative positional encoding."""
    freq_seq = tf.range(0, self.d_model, 2.0)
    if dtype is not None and dtype != tf.float32:
        freq_seq = tf.cast(freq_seq, dtype=dtype)
    inv_freq = 1 / 10000 ** (freq_seq / self.d_model)
    if self.attn_type == 'bi':
        beg, end = klen, -qlen
    elif self.attn_type == 'uni':
        beg, end = klen, -1
    else:
        raise ValueError('Unknown `attn_type` {}.'.format(self.attn_type))
    if self.bi_data:
        fwd_pos_seq = tf.range(beg, end, -1.0)
        bwd_pos_seq = tf.range(-beg, -end, 1.0)
        if dtype is not None and dtype != tf.float32:
            fwd_pos_seq = tf.cast(fwd_pos_seq, dtype=dtype)
            bwd_pos_seq = tf.cast(bwd_pos_seq, dtype=dtype)
        if self.clamp_len > 0:
            fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -self.clamp_len,
                self.clamp_len)
            bwd_pos_seq = tf.clip_by_value(bwd_pos_seq, -self.clamp_len,
                self.clamp_len)
        if bsz is not None:
            assert bsz % 2 == 0, f'With bi_data, the batch size {bsz} should be divisible by 2'
            fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, 
                bsz // 2)
            bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, 
                bsz // 2)
        else:
            fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
            bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)
        pos_emb = tf.concat([fwd_pos_emb, bwd_pos_emb], axis=1)
    else:
        fwd_pos_seq = tf.range(beg, end, -1.0)
        if dtype is not None and dtype != tf.float32:
            fwd_pos_seq = tf.cast(fwd_pos_seq, dtype=dtype)
        if self.clamp_len > 0:
            fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -self.clamp_len,
                self.clamp_len)
        pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)
    return pos_emb
