def forward(self, src_seq, src_pos, tgt_seq, tgt_pos, gold):
    out = self.transformer(src_seq, src_pos, tgt_seq, tgt_pos)
    return self._cal_loss(out, gold, smoothing=True)
